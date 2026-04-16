"""Generate a split JSON for a perturbed (cloned) dataset that mirrors the
train/val/test split of the original single-goal dataset.

Each cloned episode's metadata.json contains a ``clone_source.source_episode_stem``
field that links it back to the original episode.  This script reads the original
``episode_ids_split.json`` (produced by convert_to_tfrecord.py), then walks the
cloned dataset to assign every perturbed episode to the same split as its source.

The output JSON can be fed directly to ``convert_to_tfrecord.py --split_json``.

Usage:
    python -m gcbc_jax.generate_clone_split \
        --original_split_json  path/to/original/tfrecords/episode_ids_split.json \
        --clone_data_dir       datasets/ispatialgym-demos-d0-goal-seed1/data/task-0051 \
        --project_root         /path/to/behavior-1k-private \
        --output               clone_split.json

Multi-seed / multi-task shorthand (mirrors convert_to_tfrecord.py flags):
    python -m gcbc_jax.generate_clone_split \
        --original_split_json  path/to/original/tfrecords/episode_ids_split.json \
        --seeds 1-5 --task_id 51-53 --suffix d0-goal \
        --project_root /path/to/behavior-1k-private \
        --output clone_split.json
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd


def parse_int_list(s: str) -> list[int]:
    """Parse '1-20' (inclusive range) or '1,2,4,5' (comma-separated) into a list of ints."""
    if "-" in s and "," not in s:
        lo, hi = s.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def resolve_metadata_dir(parquet_path: str, project_root: str | None) -> str:
    """Return the instance directory that contains metadata.json for a parquet episode."""
    df = pd.read_parquet(parquet_path, columns=["image_condition_path"])
    goal_img_path = df["image_condition_path"].iloc[0]
    if not os.path.isabs(goal_img_path) and project_root:
        goal_img_path = os.path.join(project_root, goal_img_path)
    return os.path.dirname(os.path.abspath(goal_img_path))


def main():
    parser = argparse.ArgumentParser(
        description="Generate split JSON for a cloned dataset that mirrors "
                    "the original dataset's split.")
    parser.add_argument("--original_split_json", type=str, required=True,
                        help="Path to the original episode_ids_split.json")
    parser.add_argument("--clone_data_dir", type=str, default=None,
                        help="Directory containing cloned parquet files")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Seed range (e.g. 1-20) or list (e.g. 1,2,4,5)")
    parser.add_argument("--task_id", type=str, default=None,
                        help="Task ID range (e.g. 51-53) or list (e.g. 51,52)")
    parser.add_argument("--suffix", type=str, default=None,
                        help='Directory name suffix, e.g. "d0-goal"')
    parser.add_argument("--project_root", type=str, default=None)
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for the clone split JSON")
    args = parser.parse_args()

    # Validate mutually exclusive data source options
    if args.seeds is not None or args.task_id is not None:
        assert args.seeds and args.task_id, \
            "--seeds and --task_id must both be provided"
        assert args.clone_data_dir is None, \
            "--clone_data_dir cannot be used with --seeds/--task_id"
    else:
        assert args.clone_data_dir is not None, \
            "Either --clone_data_dir or --seeds + --task_id is required"

    # 1. Load original split
    with open(args.original_split_json) as f:
        original_split = json.load(f)

    source_to_split: dict[str, str] = {}
    for split_name, ep_list in original_split.items():
        for ep_id in ep_list:
            source_to_split[ep_id] = split_name
    print(f"Original split: "
          f"{sum(1 for s in source_to_split.values() if s == 'train')} train, "
          f"{sum(1 for s in source_to_split.values() if s == 'val')} val, "
          f"{sum(1 for s in source_to_split.values() if s == 'test')} test")

    # 2. Collect clone parquet paths
    parquet_paths: list[str] = []
    if args.clone_data_dir:
        parquet_paths = sorted(glob.glob(os.path.join(args.clone_data_dir, "*.parquet")))
    else:
        seeds = parse_int_list(args.seeds)
        task_ids = parse_int_list(args.task_id)
        suffix_part = f"-{args.suffix}" if args.suffix else ""
        for seed in seeds:
            for tid in task_ids:
                data_dir = f"datasets/ispatialgym-demos{suffix_part}-seed{seed}/data/task-{tid:04d}"
                parquet_paths.extend(sorted(glob.glob(os.path.join(data_dir, "*.parquet"))))

    print(f"Found {len(parquet_paths)} clone parquet files")
    if not parquet_paths:
        print("ERROR: No parquet files found.", file=sys.stderr)
        sys.exit(1)

    # 3. For each clone, look up its source episode and assign the same split
    clone_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    missing_metadata = []
    missing_source = []

    for pq_path in parquet_paths:
        ep_name = os.path.splitext(os.path.basename(pq_path))[0]
        instance_dir = resolve_metadata_dir(pq_path, args.project_root)
        meta_path = os.path.join(instance_dir, "metadata.json")

        if not os.path.exists(meta_path):
            missing_metadata.append(ep_name)
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        clone_source = meta.get("clone_source")
        if clone_source is None:
            missing_metadata.append(ep_name)
            continue

        source_stem = clone_source["source_episode_stem"]
        split = source_to_split.get(source_stem)
        if split is None:
            missing_source.append((ep_name, source_stem))
            continue

        clone_split[split].append(ep_name)

    # 4. Report and save
    if missing_metadata:
        print(f"\nWARNING: {len(missing_metadata)} episodes had no clone_source metadata:")
        for name in missing_metadata[:10]:
            print(f"  {name}")
        if len(missing_metadata) > 10:
            print(f"  ... and {len(missing_metadata) - 10} more")

    if missing_source:
        print(f"\nWARNING: {len(missing_source)} episodes' source not found in original split:")
        for name, src in missing_source[:10]:
            print(f"  {name} -> {src}")
        if len(missing_source) > 10:
            print(f"  ... and {len(missing_source) - 10} more")

    total = sum(len(v) for v in clone_split.values())
    print(f"\nClone split: {len(clone_split['train'])} train, "
          f"{len(clone_split['val'])} val, {len(clone_split['test'])} test "
          f"({total} total)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(clone_split, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
