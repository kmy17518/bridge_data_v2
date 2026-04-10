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

    # Proprioception from parquet (256-dim robot state)
    proprio_raw = np.stack(df["observation.state"].values).astype(np.float32)  # (T, 256)
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
        for seed in seeds:
            for tid in task_ids:
                data_dir = f"datasets/ispatialgym-demos-seed{seed}/data/task-{tid:04d}"
                video_dir = args.video_dir or os.path.join(
                    f"datasets/ispatialgym-demos-seed{seed}", "videos", f"task-{tid:04d}")
                for pq in sorted(glob.glob(os.path.join(data_dir, "*.parquet"))):
                    episodes.append((pq, video_dir))

    print(f"Found {len(episodes)} episodes")

    # Train/val/test split by episode (8:1:1)
    rng = np.random.RandomState(args.seed)
    indices = np.arange(len(episodes))
    rng.shuffle(indices)
    n_test = max(1, int(len(episodes) * args.test_ratio)) if args.test_ratio > 0 else 0
    n_val = max(1, int(len(episodes) * args.val_ratio)) if args.val_ratio > 0 else 0
    test_indices = set(indices[:n_test].tolist())
    val_indices = set(indices[n_test:n_test + n_val].tolist())

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

    all_actions = []
    all_proprios = []

    skipped = 0
    for i, (pq_path, video_dir) in enumerate(tqdm(episodes, desc="Converting")):
        if i in test_indices:
            split = "test"
        elif i in val_indices:
            split = "val"
        else:
            split = "train"

        ep_name = os.path.splitext(os.path.basename(pq_path))[0]
        out_path = os.path.join(args.output_dir, split, f"{ep_name}.tfrecord")

        if args.resume and os.path.exists(out_path):
            # Already converted — still collect stats from parquet (cheap)
            if split == "train":
                df = pd.read_parquet(pq_path)
                all_actions.append(np.stack(df["action"].values).astype(np.float32)[:-1])
                all_proprios.append(np.stack(df["observation.state"].values).astype(np.float32)[:-1])
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

        print(f"  [{split}] {ep_name}: {len(actions)} transitions, goal {goal_img.shape}")

    # Compute and save action + proprio normalization stats
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

    import json
    stats_path = os.path.join(args.output_dir, "action_proprio_metadata.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")
    print(f"Action mean: {action_mean[:5]}...")
    print(f"Action std:  {action_std[:5]}...")
    print(f"Proprio dim: {proprio_mean.shape[0]}, mean[:5]: {proprio_mean[:5]}...")
    print(f"Proprio std[:5]: {proprio_std[:5]}...")
    n_train = len(episodes) - n_val - n_test
    if skipped:
        print(f"\nResumed: skipped {skipped} existing episodes")
    print(f"Done. Train: {n_train}, Val: {n_val}, Test: {n_test} episodes")


if __name__ == "__main__":
    main()
