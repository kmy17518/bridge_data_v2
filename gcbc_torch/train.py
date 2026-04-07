"""Train GC-BC / GC-DDPM-BC using PyTorch with custom ISpatialGym data.

Supports both GCBC (deterministic) and DDPM diffusion policies.

Usage:
    # GCBC (default)
    python -m gcbc_torch.train \
        --tfrecord_dir gcbc_jax/tfrecords/task-0053-final \
        --save_dir outputs/gcbc_torch_task0053 \
        --num_steps 50000

    # Diffusion policy
    python -m gcbc_torch.train \
        --policy gc_ddpm_bc \
        --tfrecord_dir gcbc_jax/tfrecords/task-0053-final \
        --save_dir outputs/ddpm_torch_task0053 \
        --num_steps 50000
"""

import argparse
import copy
import glob
import json
import math
import os
import shutil

import numpy as np
import tensorflow as tf
import torch
import tqdm

from .dataset import build_tf_dataset, load_raw_trajectories, tf_batch_to_torch, tf_batch_to_torch_iql
from .diffusion_model import GCDDPMBCPolicy
from .iql_model import GCIQLPolicy
from .model import GCBCPolicy
from .vis import visualize_predictions


def get_lr_schedule(optimizer, warmup_steps, decay_steps, peak_lr):
    """Warmup + cosine decay schedule matching optax.warmup_cosine_decay_schedule.

    init_value=0, peak_value=peak_lr, end_value=0.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup: 0 -> 1
            return step / max(warmup_steps, 1)
        else:
            # Cosine decay: 1 -> 0
            progress = (step - warmup_steps) / max(decay_steps - warmup_steps, 1)
            progress = min(progress, 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args):
    # Prevent TF from using GPU (PyTorch needs it)
    tf.config.set_visible_devices([], "GPU")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Load action normalization stats
    stats_path = os.path.join(args.tfrecord_dir, "action_proprio_metadata.json")
    with open(stats_path) as f:
        action_proprio_metadata = json.load(f)
    for key in action_proprio_metadata:
        for stat in action_proprio_metadata[key]:
            action_proprio_metadata[key][stat] = np.array(
                action_proprio_metadata[key][stat], dtype=np.float32)

    # Save metadata alongside checkpoint for eval
    shutil.copy(stats_path, os.path.join(args.save_dir, "action_proprio_metadata.json"))

    # Find TFRecord paths
    train_paths = sorted(glob.glob(os.path.join(args.tfrecord_dir, "train", "*.tfrecord")))
    val_paths = sorted(glob.glob(os.path.join(args.tfrecord_dir, "val", "*.tfrecord")))
    print(f"Train TFRecords: {len(train_paths)}, Val TFRecords: {len(val_paths)}")

    # Augmentation kwargs (matching JAX exactly)
    augment_kwargs = dict(
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
    )

    # Build TF datasets for data loading
    train_dataset = build_tf_dataset(
        train_paths, args.batch_size, args.seed, train=True,
        action_proprio_metadata=action_proprio_metadata,
        augment=args.augment, augment_kwargs=augment_kwargs,
        use_proprio=args.use_proprio, add_eef_proprio=args.add_eef_proprio,
        normalize_proprio=args.normalize_proprio,
    )

    val_dataset = build_tf_dataset(
        val_paths, args.batch_size, args.seed + 1, train=False,
        action_proprio_metadata=action_proprio_metadata,
        augment=False, augment_kwargs=None,
        use_proprio=args.use_proprio, add_eef_proprio=args.add_eef_proprio,
        normalize_proprio=args.normalize_proprio,
    )

    # Load vis trajectories
    vis_trajs = load_raw_trajectories(val_paths, n=3, seed=args.seed)
    print(f"Loaded {len(vis_trajs)} val trajectories for visualization")

    # Resolve training mode (epoch vs step)
    epoch_mode = args.num_epochs is not None
    if epoch_mode:
        num_epochs = args.num_epochs
        steps_per_epoch = 0
        for _ in train_dataset:
            steps_per_epoch += 1
        total_steps = num_epochs * steps_per_epoch
        log_interval = args.log_interval * steps_per_epoch
        eval_interval = (args.eval_interval or 5) * steps_per_epoch
        save_interval = (args.save_interval or 5) * steps_per_epoch
        print(f"\nEpoch mode: {num_epochs} epochs x {steps_per_epoch} steps/epoch "
              f"= {total_steps} total steps")
        print(f"  log every {args.log_interval} epoch(s), eval every {args.eval_interval or 5} "
              f"epoch(s), save every {args.save_interval or 5} epoch(s)")
        # Rebuild train dataset since we consumed it counting steps
        train_dataset = build_tf_dataset(
            train_paths, args.batch_size, args.seed, train=True,
            action_proprio_metadata=action_proprio_metadata,
            augment=args.augment, augment_kwargs=augment_kwargs,
            use_proprio=args.use_proprio, add_eef_proprio=args.add_eef_proprio,
            normalize_proprio=args.normalize_proprio,
        )
    else:
        total_steps = args.num_steps or 100000
        log_interval = args.log_interval
        eval_interval = args.eval_interval or 5000
        save_interval = args.save_interval or 5000
        print(f"\nStep mode: {total_steps} total steps")

    # Get example batch for model init
    train_iter = iter(train_dataset.as_numpy_iterator())
    example_batch_np = next(train_iter)
    example_batch = tf_batch_to_torch(example_batch_np, device)
    action_dim = example_batch["actions"].shape[-1]
    print(f"Action dim: {action_dim}")
    print(f"Obs image shape: {example_batch['obs_image'].shape}")
    print(f"Goal image shape: {example_batch['goal_image'].shape}")

    if args.use_proprio:
        proprio_dim = example_batch["obs_proprio"].shape[-1]
        print(f"Proprio dim: {proprio_dim} (add_eef={args.add_eef_proprio}, "
              f"normalize={args.normalize_proprio})")
    else:
        proprio_dim = 23

    # Create model
    if args.policy == "gc_iql":
        model = GCIQLPolicy(
            action_dim=action_dim,
            use_proprio=args.use_proprio,
            proprio_dim=proprio_dim if args.use_proprio else 23,
            hidden_dims=(256, 256, 256),
            dropout_rate=0.1,
            discount=args.discount,
            expectile=args.expectile,
            temperature=args.temperature,
            negative_proportion=args.negative_proportion,
        ).to(device)
    elif args.policy == "gc_ddpm_bc":
        model = GCDDPMBCPolicy(
            action_dim=action_dim,
            use_proprio=args.use_proprio,
            proprio_dim=proprio_dim if args.use_proprio else 23,
            diffusion_steps=args.diffusion_steps,
            beta_schedule="cosine",
            time_dim=32,
            num_blocks=3,
            hidden_dim=256,
            dropout_rate=0.1,
            use_layer_norm=True,
        ).to(device)
    else:
        model = GCBCPolicy(
            action_dim=action_dim,
            use_proprio=args.use_proprio,
            proprio_dim=proprio_dim if args.use_proprio else 23,
            hidden_dims=(256, 256, 256),
            dropout_rate=0.1,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} (policy={args.policy})")

    # EMA target network
    target_state_dict = None
    target_model = None
    if args.policy == "gc_ddpm_bc":
        target_state_dict = copy.deepcopy(model.state_dict())
    elif args.policy == "gc_iql":
        target_model = copy.deepcopy(model)
        target_model.eval()
        for p in target_model.parameters():
            p.requires_grad_(False)

    # Optimizer: Adam with warmup + cosine decay (matching JAX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_lr_schedule(optimizer, args.warmup_steps, total_steps, args.lr)

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        ckpt_files = sorted(glob.glob(os.path.join(args.save_dir, "checkpoint_*.pt")))
        if ckpt_files:
            ckpt_path = max(ckpt_files,
                            key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if target_state_dict is not None and "target_state_dict" in ckpt:
                target_state_dict = ckpt["target_state_dict"]
            if target_model is not None and "target_state_dict" in ckpt:
                target_model.load_state_dict(ckpt["target_state_dict"])
            start_step = ckpt["step"]
            print(f"Resumed from {ckpt_path} at step {start_step}")

    # Optional WandB
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args),
                   resume="allow" if args.resume else None)

    # Training loop
    print("Starting training...")
    torch.manual_seed(args.seed)
    model.train()

    for i in tqdm.tqdm(range(start_step, total_steps)):
        try:
            batch_np = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataset.as_numpy_iterator())
            batch_np = next(train_iter)

        if args.policy == "gc_iql":
            batch = tf_batch_to_torch_iql(batch_np, device)
        else:
            batch = tf_batch_to_torch(batch_np, device)

        # Move to GPU
        if args.policy == "gc_iql":
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, metrics = model.compute_loss(batch, target_model)
        else:
            obs_image = batch["obs_image"].to(device)
            goal_image = batch["goal_image"].to(device)
            actions = batch["actions"].to(device)
            proprio = batch["obs_proprio"].to(device) if args.use_proprio else None
            loss, metrics = model.compute_loss(obs_image, goal_image, actions, proprio)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # EMA target network update
        if target_state_dict is not None:
            tau = args.target_update_rate
            with torch.no_grad():
                for k in target_state_dict:
                    target_state_dict[k].lerp_(model.state_dict()[k], tau)
        if target_model is not None:
            tau = args.target_update_rate
            with torch.no_grad():
                for tp, mp in zip(target_model.parameters(), model.parameters()):
                    tp.lerp_(mp, tau)

        step = i + 1
        metrics["lr"] = scheduler.get_last_lr()[0]

        # Logging
        if step % log_interval == 0:
            if args.use_wandb:
                import wandb
                wandb.log({f"training/{k}": v for k, v in metrics.items()}, step=step)
            loss_key = ("ddpm_loss" if args.policy == "gc_ddpm_bc"
                        else "total_loss" if args.policy == "gc_iql"
                        else "actor_loss")
            loss_val = metrics[loss_key]
            mse_val = metrics["mse"]
            if epoch_mode:
                epoch = step // steps_per_epoch
                print(f"Epoch {epoch}/{num_epochs} (step {step}): "
                      f"loss={loss_val:.4f} mse={mse_val:.4f}", flush=True)
            elif step % (log_interval * 10) == 0:
                print(f"Step {step}: loss={loss_val:.4f} "
                      f"mse={mse_val:.4f}", flush=True)

        # Validation
        if step % eval_interval == 0:
            model.eval()
            val_metrics_list = []
            for val_batch_np in val_dataset.as_numpy_iterator():
                if args.policy == "gc_iql":
                    val_batch = tf_batch_to_torch_iql(val_batch_np, device)
                else:
                    val_batch = tf_batch_to_torch(val_batch_np, device)
                with torch.no_grad():
                    if args.policy == "gc_iql":
                        val_batch = {k: v.to(device) for k, v in val_batch.items()}
                        _, v_metrics = model.compute_loss(val_batch, target_model)
                    else:
                        v_obs = val_batch["obs_image"].to(device)
                        v_goal = val_batch["goal_image"].to(device)
                        v_actions = val_batch["actions"].to(device)
                        v_proprio = val_batch["obs_proprio"].to(device) if args.use_proprio else None
                        _, v_metrics = model.compute_loss(v_obs, v_goal, v_actions, v_proprio)
                    val_metrics_list.append(v_metrics)

            if val_metrics_list:
                val_summary = {}
                for k in val_metrics_list[0]:
                    val_summary[k] = np.mean([m[k] for m in val_metrics_list])
                print(f"  Val step {step}: {val_summary}", flush=True)
                if args.use_wandb:
                    import wandb
                    wandb.log({f"validation/{k}": v for k, v in val_summary.items()},
                              step=step)

            # Visualization
            if vis_trajs:
                visualize_predictions(
                    model, vis_trajs, step=step,
                    save_dir=args.save_dir,
                    action_metadata=action_proprio_metadata,
                    device=device,
                    use_wandb=args.use_wandb,
                    use_proprio=args.use_proprio,
                    add_eef=args.add_eef_proprio,
                    normalize_proprio=args.normalize_proprio,
                    target_state_dict=target_state_dict,
                )

            model.train()

        # Checkpointing
        if step % save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_{step}.pt")
            ckpt_data = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": vars(args),
            }
            if target_state_dict is not None:
                ckpt_data["target_state_dict"] = target_state_dict
            if target_model is not None:
                ckpt_data["target_state_dict"] = target_model.state_dict()
            torch.save(ckpt_data, ckpt_path)
            print(f"  Checkpoint saved at step {step}", flush=True)

    # Final checkpoint
    ckpt_path = os.path.join(args.save_dir, f"checkpoint_{total_steps}.pt")
    ckpt_data = {
        "step": total_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
    }
    if target_state_dict is not None:
        ckpt_data["target_state_dict"] = target_state_dict
    if target_model is not None:
        ckpt_data["target_state_dict"] = target_model.state_dict()
    torch.save(ckpt_data, ckpt_path)
    print(f"\nTraining complete. Checkpoints in {args.save_dir}")

    if args.use_wandb:
        import wandb
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train GC-BC / GC-DDPM-BC (PyTorch)")
    parser.add_argument("--policy", type=str, default="gcbc",
                        choices=["gcbc", "gc_ddpm_bc", "gc_iql"],
                        help="Policy type: gcbc, gc_ddpm_bc (diffusion), or gc_iql")
    parser.add_argument("--tfrecord_dir", type=str, required=True,
                        help="Directory with train/val TFRecords + action_proprio_metadata.json")
    parser.add_argument("--save_dir", type=str, default="outputs/gcbc_torch")
    parser.add_argument("--run_name", type=str, default="gcbc_torch")
    parser.add_argument("--encoder", type=str, default="resnetv1-34-bridge")

    parser.add_argument("--num_steps", type=int, default=None,
                        help="Total training steps (step mode, default)")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Total training epochs (epoch mode)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", dest="augment", action="store_false")

    # Diffusion-specific args
    parser.add_argument("--diffusion_steps", type=int, default=20,
                        help="Number of diffusion timesteps (gc_ddpm_bc only)")
    parser.add_argument("--target_update_rate", type=float, default=0.002,
                        help="EMA target network update rate (gc_ddpm_bc / gc_iql)")

    # IQL-specific args
    parser.add_argument("--discount", type=float, default=0.98,
                        help="Discount factor (gc_iql only)")
    parser.add_argument("--expectile", type=float, default=0.7,
                        help="Expectile for value loss (gc_iql only)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for advantage weighting (gc_iql only)")
    parser.add_argument("--negative_proportion", type=float, default=0.1,
                        help="Proportion of negative goals (gc_iql only)")

    parser.add_argument("--use_proprio", action="store_true",
                        help="Use 23-dim proprio (base_qvel, trunk, arms, grippers)")
    parser.add_argument("--add_eef_proprio", action="store_true",
                        help="Extend to 37-dim by adding EEF pos+quat (requires --use_proprio)")
    parser.add_argument("--normalize_proprio", action="store_true",
                        help="Normalize proprio to [-1,1] using JOINT_RANGE bounds")

    parser.add_argument("--log_interval", type=int, default=100,
                        help="Log every N steps (step mode) or N epochs (epoch mode)")
    parser.add_argument("--eval_interval", type=int, default=None,
                        help="Eval every N steps/epochs (default: 5000 steps or 5 epochs)")
    parser.add_argument("--save_interval", type=int, default=None,
                        help="Save every N steps/epochs (default: 5000 steps or 5 epochs)")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="gcbc-ispatialgym")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in save_dir")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
