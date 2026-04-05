"""Train GC-BC using the original bridge_data_v2 agent with custom ISpatialGym data.

Usage:
    python -m gcbc_jax.train \
        --tfrecord_dir gcbc_jax/tfrecords/task-0051 \
        --save_dir outputs/gcbc_jax_task0051 \
        --num_steps 100000
"""

import argparse
import glob
import json
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm
from flax import serialization

from jaxrl_m.agents.continuous.gc_bc import GCBCAgent
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.vision import encoders

from .dataset import FixedGoalBridgeDataset
from .vis import load_vis_trajectories, visualize_predictions


def shard_batch(batch, sharding):
    """Distribute batch across devices."""
    return jax.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


def train(args):
    # Prevent TF from using GPU (JAX needs it)
    tf.config.set_visible_devices([], "GPU")

    # Device setup
    devices = jax.local_devices()
    num_devices = len(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    print(f"JAX devices: {devices}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Load action normalization stats
    stats_path = os.path.join(args.tfrecord_dir, "action_proprio_metadata.json")
    with open(stats_path) as f:
        action_proprio_metadata = json.load(f)
    # Convert to numpy
    for key in action_proprio_metadata:
        for stat in action_proprio_metadata[key]:
            action_proprio_metadata[key][stat] = np.array(
                action_proprio_metadata[key][stat], dtype=np.float32
            )

    # Save metadata alongside checkpoint for eval
    import shutil
    shutil.copy(stats_path, os.path.join(args.save_dir, "action_proprio_metadata.json"))

    # Find TFRecord paths
    train_paths = sorted(glob.glob(os.path.join(args.tfrecord_dir, "train", "*.tfrecord")))
    val_paths = sorted(glob.glob(os.path.join(args.tfrecord_dir, "val", "*.tfrecord")))
    print(f"Train TFRecords: {len(train_paths)}, Val TFRecords: {len(val_paths)}")

    # Dataset kwargs (from original gc_bc config)
    dataset_kwargs = dict(
        goal_relabeling_strategy="uniform",  # overridden by FixedGoalBridgeDataset
        goal_relabeling_kwargs=dict(reached_proportion=0.0),
        relabel_actions=False,  # our actions are already correct
        shuffle_buffer_size=25000,
        augment=args.augment,
        augment_kwargs=dict(
            random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
            random_brightness=[0.2],
            random_contrast=[0.8, 1.2],
            random_saturation=[0.8, 1.2],
            random_hue=[0.1],
            augment_order=[
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ],
        ),
    )

    proprio_kwargs = dict(
        use_proprio=args.use_proprio,
        add_eef_proprio=args.add_eef_proprio,
        normalize_proprio=args.normalize_proprio,
    )

    train_data = FixedGoalBridgeDataset(
        [train_paths],
        seed=args.seed,
        batch_size=args.batch_size,
        train=True,
        action_proprio_metadata=action_proprio_metadata,
        **proprio_kwargs,
        **dataset_kwargs,
    )

    val_data = FixedGoalBridgeDataset(
        [val_paths],
        seed=args.seed + 1,
        batch_size=args.batch_size,
        train=False,
        action_proprio_metadata=action_proprio_metadata,
        **proprio_kwargs,
        **dataset_kwargs,
    )

    # Load 3 fixed val trajectories for visualization
    vis_trajs = load_vis_trajectories(val_paths, n=3, seed=args.seed)
    print(f"Loaded {len(vis_trajs)} val trajectories for visualization")
    vis_rng = jax.random.PRNGKey(args.seed + 100)

    train_data_iter = map(shard_fn, train_data.tf_dataset.as_numpy_iterator())

    # Get example batch for agent initialization
    example_batch = next(train_data_iter)
    action_dim = example_batch["actions"].shape[-1]
    print(f"Action dim: {action_dim}")
    print(f"Obs image shape: {example_batch['observations']['image'].shape}")
    print(f"Goal image shape: {example_batch['goals']['image'].shape}")
    if args.use_proprio:
        proprio_dim = example_batch["observations"]["proprio"].shape[-1]
        print(f"Proprio dim: {proprio_dim} (add_eef={args.add_eef_proprio}, "
              f"normalize={args.normalize_proprio})")

    # Create encoder (original resnetv1-34-bridge)
    encoder_def = encoders[args.encoder](
        pooling_method="avg",
        add_spatial_coordinates=True,
        act="swish",
    )

    # Create agent (original GCBCAgent)
    rng = jax.random.PRNGKey(args.seed)
    rng, construct_rng = jax.random.split(rng)

    agent = GCBCAgent.create(
        rng=construct_rng,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        encoder_def=encoder_def,
        early_goal_concat=True,
        shared_goal_encoder=True,
        use_proprio=args.use_proprio,
        network_kwargs=dict(hidden_dims=(256, 256, 256), dropout_rate=0.1),
        policy_kwargs=dict(
            tanh_squash_distribution=False,
            fixed_std=[1.0] * action_dim,
            state_dependent_std=False,
        ),
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        decay_steps=args.num_steps,
    )

    # Replicate across devices
    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(agent.state.params))
    print(f"Model parameters: {n_params:,}")

    # Optional WandB
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # Training loop
    print(f"\nStarting training for {args.num_steps} steps...")
    for i in tqdm.tqdm(range(int(args.num_steps))):
        try:
            batch = next(train_data_iter)
        except StopIteration:
            train_data_iter = map(shard_fn, train_data.tf_dataset.as_numpy_iterator())
            batch = next(train_data_iter)

        agent, update_info = agent.update(batch)

        # Logging
        if (i + 1) % args.log_interval == 0:
            info = jax.device_get(update_info)
            log_dict = {}
            for k, v in info.items():
                try:
                    log_dict[k] = float(v)
                except (TypeError, ValueError):
                    pass
            if args.use_wandb:
                import wandb
                wandb.log({f"training/{k}": v for k, v in log_dict.items()}, step=i + 1)
            if (i + 1) % (args.log_interval * 10) == 0:
                print(f"Step {i+1}: loss={log_dict.get('actor_loss', 'N/A'):.4f} "
                      f"mse={log_dict.get('mse', 'N/A'):.4f}", flush=True)

        # Validation
        if (i + 1) % args.eval_interval == 0:
            val_metrics = []
            val_data_iter = map(shard_fn, val_data.iterator())
            for val_batch in val_data_iter:
                rng, val_rng = jax.random.split(rng)
                val_metrics.append(agent.get_debug_metrics(val_batch, seed=val_rng))
            if val_metrics:
                val_summary = jax.tree_map(lambda *xs: np.mean(xs), *val_metrics)
                val_log = {}
                for k, v in val_summary.items():
                    try:
                        val_log[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                print(f"  Val step {i+1}: {val_log}", flush=True)
                if args.use_wandb:
                    import wandb
                    wandb.log({f"validation/{k}": v for k, v in val_log.items()}, step=i + 1)

            # Visualization on fixed val trajectories
            if vis_trajs:
                vis_rng = visualize_predictions(
                    agent, vis_trajs, step=i + 1,
                    save_dir=args.save_dir,
                    action_metadata=action_proprio_metadata,
                    rng=vis_rng,
                    use_wandb=args.use_wandb,
                    use_proprio=args.use_proprio,
                    add_eef=args.add_eef_proprio,
                    normalize_proprio=args.normalize_proprio,
                )

        # Checkpointing (flax msgpack, avoids orbax version issues)
        if (i + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_{i + 1}")
            agent_cpu = jax.device_get(jax.tree_map(np.array, agent))
            with open(ckpt_path, "wb") as f:
                f.write(serialization.to_bytes(agent_cpu))
            print(f"  Checkpoint saved at step {i+1}", flush=True)

    # Final checkpoint
    ckpt_path = os.path.join(args.save_dir, f"checkpoint_{int(args.num_steps)}")
    agent_cpu = jax.device_get(jax.tree_map(np.array, agent))
    with open(ckpt_path, "wb") as f:
        f.write(serialization.to_bytes(agent_cpu))
    print(f"\nTraining complete. Checkpoints in {args.save_dir}")

    if args.use_wandb:
        import wandb
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train GC-BC (JAX)")
    parser.add_argument("--tfrecord_dir", type=str, required=True,
                        help="Directory with train/val TFRecords + action_proprio_metadata.json")
    parser.add_argument("--save_dir", type=str, default="outputs/gcbc_jax")
    parser.add_argument("--run_name", type=str, default="gcbc_jax")
    parser.add_argument("--encoder", type=str, default="resnetv1-34-bridge")

    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", dest="augment", action="store_false")

    parser.add_argument("--use_proprio", action="store_true",
                        help="Use 23-dim proprio (base_qvel, trunk, arms, grippers)")
    parser.add_argument("--add_eef_proprio", action="store_true",
                        help="Extend to 37-dim by adding EEF pos+quat (requires --use_proprio)")
    parser.add_argument("--normalize_proprio", action="store_true",
                        help="Normalize proprio to [-1,1] using JOINT_RANGE bounds")

    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="gcbc-ispatialgym")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
