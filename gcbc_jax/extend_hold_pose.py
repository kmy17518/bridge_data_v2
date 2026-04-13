"""Extend hold-pose actions at the end of TFRecord episodes.

Reads TFRecords from an input directory, appends extra hold-pose transitions
to reach a target count, and writes to a new output directory.

Usage:
    python -m gcbc_jax.extend_hold_pose \
        --input_dir gcbc_jax/tfrecords/single-goal-image/task-0054 \
        --output_dir gcbc_jax/tfrecords/single-goal-image-hold50/task-0054 \
        --target_hold_count 50
"""

import argparse
import json
import os
import shutil

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def count_hold_pose(actions, atol=1e-6):
    """Count identical trailing actions (hold-pose actions at the end)."""
    last = actions[-1]
    count = 0
    for i in range(actions.shape[0] - 1, -1, -1):
        if np.allclose(actions[i], last, atol=atol):
            count += 1
        else:
            break
    return count


def extend_episode(record_bytes, target_hold_count):
    """Extend one TFRecord episode to have target_hold_count hold-pose steps.

    Returns (new_record_bytes, actions, obs_state, was_extended).
    actions and obs_state are returned for stats collection (from the extended episode).
    """
    example = tf.train.Example()
    example.ParseFromString(record_bytes)

    feat = example.features.feature

    # Deserialize fields
    actions = tf.io.parse_tensor(
        feat["actions"].bytes_list.value[0], out_type=tf.float32).numpy()
    obs_state = tf.io.parse_tensor(
        feat["observations/state"].bytes_list.value[0], out_type=tf.float32).numpy()
    next_obs_state = tf.io.parse_tensor(
        feat["next_observations/state"].bytes_list.value[0], out_type=tf.float32).numpy()
    terminals = tf.io.parse_tensor(
        feat["terminals"].bytes_list.value[0], out_type=tf.bool).numpy()
    truncates = tf.io.parse_tensor(
        feat["truncates"].bytes_list.value[0], out_type=tf.bool).numpy()
    obs_images = tf.io.parse_tensor(
        feat["observations/images0"].bytes_list.value[0], out_type=tf.string)
    next_obs_images = tf.io.parse_tensor(
        feat["next_observations/images0"].bytes_list.value[0], out_type=tf.string)
    goal_image_bytes = feat["goal_image"].bytes_list.value[0]

    current_hold = count_hold_pose(actions)
    if current_hold >= target_hold_count:
        return record_bytes, actions, obs_state, False

    extra = target_hold_count - current_hold

    # Extend actions: repeat last action
    extra_actions = np.tile(actions[-1:], (extra, 1))
    new_actions = np.concatenate([actions, extra_actions], axis=0)

    # Extend states: repeat last next_obs_state
    last_state = next_obs_state[-1:]  # (1, 256)
    extra_obs_state = np.tile(last_state, (extra, 1))
    extra_next_obs_state = np.tile(last_state, (extra, 1))
    new_obs_state = np.concatenate([obs_state, extra_obs_state], axis=0)
    new_next_obs_state = np.concatenate([next_obs_state, extra_next_obs_state], axis=0)

    # Extend images: repeat last next_obs image JPEG bytes
    last_jpeg = next_obs_images[-1]  # scalar tf.string
    extra_jpegs = tf.fill([extra], last_jpeg)
    new_obs_images = tf.concat([obs_images, extra_jpegs], axis=0)
    new_next_obs_images = tf.concat([next_obs_images, extra_jpegs], axis=0)

    # Extend terminals: old last -> False, new last -> True
    new_terminals = np.zeros(len(new_actions), dtype=bool)
    new_terminals[-1] = True

    # Extend truncates: all False
    new_truncates = np.zeros(len(new_actions), dtype=bool)

    # Serialize everything
    def ser(tensor):
        return tf.io.serialize_tensor(tf.constant(tensor)).numpy()

    def ser_tf(tensor):
        return tf.io.serialize_tensor(tensor).numpy()

    new_feat = {
        "observations/images0": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[ser_tf(new_obs_images)])),
        "observations/state": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[ser(new_obs_state)])),
        "next_observations/images0": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[ser_tf(new_next_obs_images)])),
        "next_observations/state": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[ser(new_next_obs_state)])),
        "actions": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[ser(new_actions)])),
        "terminals": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[ser(new_terminals)])),
        "truncates": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[ser(new_truncates)])),
        "goal_image": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[goal_image_bytes])),
    }
    new_example = tf.train.Example(
        features=tf.train.Features(feature=new_feat))
    return new_example.SerializeToString(), new_actions, new_obs_state, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_hold_count", type=int, default=50)
    args = parser.parse_args()

    # Create output split directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)

    # Collect all tfrecord files
    all_files = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(args.input_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in sorted(os.listdir(split_dir)):
            if fname.endswith(".tfrecord"):
                all_files.append((split, fname))

    print(f"Found {len(all_files)} TFRecord files")
    print(f"Target hold-pose count: {args.target_hold_count}")

    # Process each file
    all_train_actions = []
    all_train_proprios = []
    extended_count = 0

    for split, fname in tqdm(all_files, desc="Extending"):
        in_path = os.path.join(args.input_dir, split, fname)
        out_path = os.path.join(args.output_dir, split, fname)

        raw = tf.data.TFRecordDataset(in_path)
        for record in raw:
            new_bytes, actions, obs_state, was_extended = extend_episode(
                record.numpy(), args.target_hold_count)
            if was_extended:
                extended_count += 1
            break

        with tf.io.TFRecordWriter(out_path) as writer:
            writer.write(new_bytes)

        if split == "train":
            all_train_actions.append(actions)
            all_train_proprios.append(obs_state)

    # Recompute normalization stats from train episodes
    all_train_actions = np.concatenate(all_train_actions, axis=0)
    action_mean = all_train_actions.mean(axis=0)
    action_std = np.clip(all_train_actions.std(axis=0), a_min=1e-3, a_max=None)

    all_train_proprios = np.concatenate(all_train_proprios, axis=0)
    proprio_mean = all_train_proprios.mean(axis=0)
    proprio_std = np.clip(all_train_proprios.std(axis=0), a_min=1e-3, a_max=None)

    # Determine image encoding from source metadata
    src_meta_path = os.path.join(args.input_dir, "action_proprio_metadata.json")
    image_encoding = "jpeg"
    if os.path.exists(src_meta_path):
        with open(src_meta_path) as f:
            src_meta = json.load(f)
        image_encoding = src_meta.get("image_encoding", "jpeg")

    stats = {
        "action": {"mean": action_mean.tolist(), "std": action_std.tolist()},
        "proprio": {"mean": proprio_mean.tolist(), "std": proprio_std.tolist()},
        "image_encoding": image_encoding,
    }
    stats_path = os.path.join(args.output_dir, "action_proprio_metadata.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Copy auxiliary files
    for aux_file in ["episode_ids_split.json", "test_episodes.json"]:
        src = os.path.join(args.input_dir, aux_file)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, aux_file))

    print(f"\nDone. Extended {extended_count}/{len(all_files)} episodes")
    print(f"Stats saved to {stats_path}")
    print(f"Action mean[:5]: {action_mean[:5]}")
    print(f"Action std[:5]:  {action_std[:5]}")


if __name__ == "__main__":
    main()
