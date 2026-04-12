"""Generate episode_ids_split.json from an existing tfrecords directory.

Given a directory with train/, val/, test/ subdirectories containing .tfrecord
files, produces a JSON file mapping each split to its episode IDs. The output
is compatible with convert_to_tfrecord.py --split_json.

Usage:
    python -m gcbc_jax.make_split_json /path/to/tfrecords/task-0053
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tfrecords_dir", type=str,
                        help="Path to tfrecords directory containing train/val/test folders")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: <tfrecords_dir>/episode_ids_split.json)")
    args = parser.parse_args()

    split_by_name = {}
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(args.tfrecords_dir, split)
        if not os.path.isdir(split_dir):
            continue
        episodes = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(split_dir)
            if f.endswith(".tfrecord")
        )
        split_by_name[split] = episodes

    total = sum(len(v) for v in split_by_name.values())
    out_path = args.output or os.path.join(args.tfrecords_dir, "episode_ids_split.json")
    with open(out_path, "w") as f:
        json.dump(split_by_name, f, indent=2)

    for split, eps in split_by_name.items():
        print(f"  {split}: {len(eps)} episodes")
    print(f"Total: {total} episodes")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
