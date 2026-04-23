"""Convert ISpatialGym parquet + video data to BridgeDataset TFRecord format.

Produces one TFRecord per episode containing a single trajectory, matching the
format expected by jaxrl_m.data.bridge_dataset.BridgeDataset.

Additionally stores the goal reference image in a separate field so that
our custom goal relabeling can use it instead of future-state sampling.

Usage:
    python -m gcbc_jax.convert_to_tfrecord \
        --data_dir datasets/ispatialgym-eval-demos/data/task-0051 \
        --output_dir gcbc_jax/tfrecords/task-0051 \
        --project_root /path/to/behavior-1k-private
"""

import argparse
import glob
import json
import os

import av
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


def decode_video_frames(video_path: str) -> np.ndarray:
    """Decode all RGB frames from an MP4. Returns (T, H, W, 3) uint8."""
    container = av.open(video_path)
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    container.close()
    return np.stack(frames)


def serialize_tensor(tensor):
    """Serialize a numpy array as a TF tensor bytes feature."""
    t = tf.constant(tensor)
    return tf.io.serialize_tensor(t).numpy()


def serialize_jpeg_frames(frames, quality=95):
    """JPEG-encode each frame and serialize as a 1-D string tensor.

    Args:
        frames: (T, H, W, 3) uint8 numpy array.
        quality: JPEG quality (1-100). 95 gives <2/255 error per pixel.

    Returns:
        bytes: serialized 1-D tf.string tensor of JPEG byte strings.
    """
    jpeg_list = [
        tf.io.encode_jpeg(tf.constant(frames[i]), quality=quality)
        for i in range(frames.shape[0])
    ]
    return tf.io.serialize_tensor(tf.stack(jpeg_list)).numpy()


def serialize_jpeg_single(image, quality=95):
    """JPEG-encode a single image and serialize as a scalar string tensor."""
    jpeg = tf.io.encode_jpeg(tf.constant(image), quality=quality)
    return tf.io.serialize_tensor(jpeg).numpy()


def make_example(obs_images, next_obs_images, obs_state, next_obs_state,
                 actions, terminals, truncates, goal_image, jpeg=True):
    """Create a tf.train.Example for one trajectory.

    When jpeg=True (default), image fields are stored as JPEG-compressed
    string tensors for ~10-15x smaller files. When jpeg=False, images are
    stored as raw uint8 tensors (lossless).
    """
    if jpeg:
        ser_frames = serialize_jpeg_frames
        ser_single = serialize_jpeg_single
    else:
        ser_frames = serialize_tensor
        ser_single = serialize_tensor

    feature = {
        "observations/images0": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[ser_frames(obs_images)])),
        "observations/state": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[serialize_tensor(obs_state)])),
        "next_observations/images0": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[ser_frames(next_obs_images)])),
        "next_observations/state": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[serialize_tensor(next_obs_state)])),
        "actions": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[serialize_tensor(actions)])),
        "terminals": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[serialize_tensor(terminals)])),
        "truncates": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[serialize_tensor(truncates)])),
        "goal_image": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[ser_single(goal_image)])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def resize_frames(frames, size):
    """Resize an array of frames (T, H, W, 3) to (T, size, size, 3)."""
    resized = np.empty((frames.shape[0], size, size, 3), dtype=np.uint8)
    for i in range(frames.shape[0]):
        resized[i] = np.array(Image.fromarray(frames[i]).resize((size, size)))
    return resized


def convert_episode(parquet_path, video_dir, project_root, action_dim=23, image_size=256):
    """Convert one episode to TFRecord-compatible numpy arrays."""
    episode_name = os.path.splitext(os.path.basename(parquet_path))[0]

    # Load parquet
    df = pd.read_parquet(parquet_path)
    T = len(df)

    # Load head camera video
    vid_path = os.path.join(video_dir, "observation.images.rgb.head", f"{episode_name}.mp4")
    frames = decode_video_frames(vid_path)  # (T, H, W, 3) uint8
    assert frames.shape[0] == T, f"Frame count {frames.shape[0]} != parquet rows {T}"

    # Resize frames to target resolution
    if frames.shape[1] != image_size or frames.shape[2] != image_size:
        frames = resize_frames(frames, image_size)

    # Observation and next observation images
    obs_images = frames[:-1]       # (T-1, H, W, 3)
    next_obs_images = frames[1:]   # (T-1, H, W, 3)

    # Actions (drop last since we have T-1 transitions)
    actions_raw = np.stack(df["action"].values).astype(np.float32)
    actions = actions_raw[:-1]     # (T-1, action_dim)

    # # TMP
    # # Force gripper actions to open (+1).
    # # Gripper dims: 14 (left), 22 (right) in the 23-dim action vector.
    # actions[:, 14] = 1.0
    # actions[:, 22] = 1.0

    # Proprioception from parquet (256-dim robot state)
    proprio_raw = np.stack(df["observation.state"].values).astype(np.float32)  # (T, 256)

    # # TMP
    # # Force gripper finger joint positions to fully open (0.05).
    # # Left gripper fingers: state[193:195], right: state[232:234]. Range [0, 0.05].
    # proprio_raw[:, 193:195] = 0.05
    # proprio_raw[:, 232:234] = 0.05

    obs_state = proprio_raw[:-1]       # (T-1, 256)
    next_obs_state = proprio_raw[1:]   # (T-1, 256)

    # Terminals and truncates
    terminals = np.zeros(T - 1, dtype=bool)
    terminals[-1] = True
    truncates = np.zeros(T - 1, dtype=bool)

    # Goal reference image — resize to same resolution as observations
    goal_img_path = df["image_condition_path"].iloc[0]
    if not os.path.isabs(goal_img_path) and project_root:
        goal_img_path = os.path.join(project_root, goal_img_path)
    goal_img = np.array(Image.open(goal_img_path).convert("RGB").resize((image_size, image_size)))

    return (obs_images, next_obs_images, obs_state, next_obs_state,
            actions, terminals, truncates, goal_img, episode_name)


def parse_int_list(s: str) -> list[int]:
    """Parse '1-20' (inclusive range) or '1,2,4,5' (comma-separated) into a list of ints."""
    if "-" in s and "," not in s:
        lo, hi = s.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def resolve_instance_metadata(parquet_path, project_root):
    """Derive instance directory paths from a parquet's image_condition_path.

    Returns dict with keys: bddl_path, template_path, tro_path, goal_image_path.
    All paths are absolute.
    """
    df = pd.read_parquet(parquet_path, columns=["image_condition_path"])
    goal_img_path = df["image_condition_path"].iloc[0]
    if not os.path.isabs(goal_img_path) and project_root:
        goal_img_path = os.path.join(project_root, goal_img_path)
    goal_img_path = os.path.abspath(goal_img_path)

    instance_dir = os.path.dirname(goal_img_path)

    bddl_files = glob.glob(os.path.join(instance_dir, "bddl", "*.bddl"))
    assert len(bddl_files) == 1, f"Expected 1 BDDL file in {instance_dir}/bddl, got {len(bddl_files)}"

    template_files = glob.glob(os.path.join(instance_dir, "*_template.json"))
    assert len(template_files) == 1, f"Expected 1 template file in {instance_dir}, got {len(template_files)}"

    tro_files = glob.glob(os.path.join(instance_dir, "*-tro_state.json"))
    assert len(tro_files) == 1, f"Expected 1 tro_state file in {instance_dir}, got {len(tro_files)}"

    return {
        "bddl_path": os.path.abspath(bddl_files[0]),
        "template_path": os.path.abspath(template_files[0]),
        "tro_path": os.path.abspath(tro_files[0]),
        "goal_image_path": goal_img_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Seed range (e.g. 1-20) or list (e.g. 1,2,4,5)")
    parser.add_argument("--task_id", type=str, default=None,
                        help="Task ID range (e.g. 51-53) or list (e.g. 51,52)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--project_root", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default=None,
                        help="Video dir (auto-detected from data_dir if not set)")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Resize images to this resolution (default 256, matching bridge_data_v2)")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_loss", action="store_true",
                        help="Use lossless raw encoding for images instead of JPEG")
    parser.add_argument("--resume", action="store_true",
                        help="Skip episodes whose .tfrecord already exists")
    parser.add_argument("--split_json", type=str, default=None,
                        help='Path to JSON with {"train": [...], "val": [...], "test": [...]} episode IDs')
    parser.add_argument("--suffix", type=str, default=None,
                        help='Directory name suffix inserted before seed, e.g. "d0-goal" gives '
                             'ispatialgym-demos-d0-goal-seed{seed}')
    args = parser.parse_args()

    # Validate mutually exclusive data source options
    if args.seeds is not None or args.task_id is not None:
        assert args.seeds and args.task_id, \
            "--seeds and --task_id must both be provided"
        assert args.data_dir is None, \
            "--data_dir cannot be used with --seeds/--task_id"
        seeds = parse_int_list(args.seeds)
        task_ids = parse_int_list(args.task_id)
    else:
        assert args.data_dir is not None, \
            "Either --data_dir or --seeds + --task_id is required"

    if not args.split_json:
        assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
            f"Ratios must sum to 1.0, got {args.train_ratio + args.val_ratio + args.test_ratio}"

    # Build list of (parquet_path, video_dir) tuples
    episodes = []
    if args.data_dir:
        if args.video_dir is None:
            task_name = os.path.basename(args.data_dir)
            demos_root = os.path.dirname(os.path.dirname(args.data_dir))
            args.video_dir = os.path.join(demos_root, "videos", task_name)
        for pq in sorted(glob.glob(os.path.join(args.data_dir, "*.parquet"))):
            episodes.append((pq, args.video_dir))
    else:
        suffix_part = f"-{args.suffix}" if args.suffix else ""
        for seed in seeds:
            for tid in task_ids:
                demos_name = f"ispatialgym-demos{suffix_part}-seed{seed}"
                data_dir = f"datasets/{demos_name}/data/task-{tid:04d}"
                video_dir = args.video_dir or os.path.join(
                    f"datasets/{demos_name}", "videos", f"task-{tid:04d}")
                for pq in sorted(glob.glob(os.path.join(data_dir, "*.parquet"))):
                    episodes.append((pq, video_dir))

    print(f"Found {len(episodes)} episodes")

    # Build episode_name -> split mapping
    episode_names = [os.path.splitext(os.path.basename(pq))[0] for pq, _ in episodes]

    if args.split_json:
        with open(args.split_json) as f:
            split_spec = json.load(f)
        episode_to_split = {}
        for split_name, ep_list in split_spec.items():
            for ep_id in ep_list:
                episode_to_split[ep_id] = split_name
        # All found episodes must be covered by the split JSON
        all_split_episodes = set(episode_to_split.keys())
        found_set = set(episode_names)
        missing = found_set - all_split_episodes
        assert not missing, (
            f"{len(missing)} episodes not found in split JSON: "
            f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
        )
    else:
        # Random train/val/test split
        rng = np.random.RandomState(args.seed)
        indices = np.arange(len(episodes))
        rng.shuffle(indices)
        n_test = max(1, int(len(episodes) * args.test_ratio)) if args.test_ratio > 0 else 0
        n_val = max(1, int(len(episodes) * args.val_ratio)) if args.val_ratio > 0 else 0
        test_indices = set(indices[:n_test].tolist())
        val_indices = set(indices[n_test:n_test + n_val].tolist())
        episode_to_split = {}
        for i, name in enumerate(episode_names):
            if i in test_indices:
                episode_to_split[name] = "test"
            elif i in val_indices:
                episode_to_split[name] = "val"
            else:
                episode_to_split[name] = "train"

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

    all_actions = []
    all_proprios = []
    test_episode_metadata = {}

    skipped = 0
    for i, (pq_path, video_dir) in enumerate(tqdm(episodes, desc="Converting")):
        ep_name = os.path.splitext(os.path.basename(pq_path))[0]
        split = episode_to_split[ep_name]
        out_path = os.path.join(args.output_dir, split, f"{ep_name}.tfrecord")

        if args.resume and os.path.exists(out_path):
            # Already converted — still collect stats from parquet (cheap)
            if split == "train":
                df = pd.read_parquet(pq_path)
                all_actions.append(np.stack(df["action"].values).astype(np.float32)[:-1])
                all_proprios.append(np.stack(df["observation.state"].values).astype(np.float32)[:-1])
            elif split == "test":
                meta = resolve_instance_metadata(pq_path, args.project_root)
                meta["tfrecord_path"] = os.path.abspath(out_path)
                test_episode_metadata[ep_name] = meta
            skipped += 1
            continue

        (obs_images, next_obs_images, obs_state, next_obs_state,
         actions, terminals, truncates, goal_img, ep_name) = convert_episode(
            pq_path, video_dir, args.project_root, image_size=args.image_size)

        if split == "train":
            all_actions.append(actions)
            all_proprios.append(obs_state)

        # Write TFRecord (one trajectory per file)
        with tf.io.TFRecordWriter(out_path) as writer:
            example = make_example(
                obs_images, next_obs_images, obs_state, next_obs_state,
                actions, terminals, truncates, goal_img,
                jpeg=(not args.no_loss))
            writer.write(example.SerializeToString())

        if split == "test":
            meta = resolve_instance_metadata(pq_path, args.project_root)
            meta["tfrecord_path"] = os.path.abspath(out_path)
            test_episode_metadata[ep_name] = meta

        print(f"  [{split}] {ep_name}: {len(actions)} transitions, goal {goal_img.shape}")

    # Compute and save action + proprio normalization stats (only if train episodes exist)
    if all_actions:
        all_actions = np.concatenate(all_actions, axis=0)
        action_mean = all_actions.mean(axis=0)
        action_std = np.clip(all_actions.std(axis=0), a_min=1e-3, a_max=None)

        all_proprios = np.concatenate(all_proprios, axis=0)
        proprio_mean = all_proprios.mean(axis=0)
        proprio_std = np.clip(all_proprios.std(axis=0), a_min=1e-3, a_max=None)

        stats = {
            "action": {"mean": action_mean.tolist(), "std": action_std.tolist()},
            "proprio": {"mean": proprio_mean.tolist(), "std": proprio_std.tolist()},
            "image_encoding": "raw" if args.no_loss else "jpeg",
        }

        stats_path = os.path.join(args.output_dir, "action_proprio_metadata.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to {stats_path}")
    else:
        action_mean = action_std = proprio_mean = proprio_std = None
        print("\nNo train episodes — skipping action/proprio normalization stats")

    # Save the split assignment so it can be reused via --split_json
    split_by_name = {"train": [], "val": [], "test": []}
    for name, s in sorted(episode_to_split.items()):
        split_by_name.setdefault(s, []).append(name)
    split_json_path = os.path.join(args.output_dir, "episode_ids_split.json")
    with open(split_json_path, "w") as f:
        json.dump(split_by_name, f, indent=2)
    print(f"Split saved to {split_json_path}")

    if test_episode_metadata:
        test_episodes_path = os.path.join(args.output_dir, "test_episodes.json")
        with open(test_episodes_path, "w") as f:
            json.dump(test_episode_metadata, f, indent=2)
        print(f"Test episodes JSON saved to {test_episodes_path} ({len(test_episode_metadata)} episodes)")
    if action_mean is not None:
        print(f"Action mean: {action_mean[:5]}...")
        print(f"Action std:  {action_std[:5]}...")
        print(f"Proprio dim: {proprio_mean.shape[0]}, mean[:5]: {proprio_mean[:5]}...")
        print(f"Proprio std[:5]: {proprio_std[:5]}...")
    n_train = sum(1 for s in episode_to_split.values() if s == "train")
    n_val = sum(1 for s in episode_to_split.values() if s == "val")
    n_test = sum(1 for s in episode_to_split.values() if s == "test")
    if skipped:
        print(f"\nResumed: skipped {skipped} existing episodes")
    print(f"Done. Train: {n_train}, Val: {n_val}, Test: {n_test} episodes")


if __name__ == "__main__":
    main()
