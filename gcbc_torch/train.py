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
import csv
import glob
import json
import math
import os

import numpy as np
import tensorflow as tf
import torch
import tqdm

from .dataset import build_tf_dataset, load_raw_trajectories, tf_batch_to_torch, tf_batch_to_torch_iql
from .diffusion_model import GCDDPMBCPolicy
from .iql_model import GCIQLPolicy
from .model import GCBCPolicy
from .proprio import (
    ABLATION_MODES,
    ablation_uses_image,
    ablation_uses_proprio,
    ablation_proprio_dim,
    apply_ablation_to_proprio,
    parse_base_proprio_keys,
)
from .vis import visualize_predictions


def _append_metrics_csv(csv_path, step, train_loss=None, val_loss=None, val_l1=None):
    """Append one row to a local ``logs/metrics.csv``.

    Mirrors the schema written by ACT's Lightning CSVLogger (and
    ``fetch_wandb_metrics.py``) so ``build_all_checkpoints_json.py`` can read
    GCBC train/val loss the same way it reads ACT's -- no wandb round-trip
    needed for the eval plots. Train rows (logged frequently) and val rows
    (logged at eval cadence) are written separately, leaving the other columns
    blank; the builder collects each metric series independently. Creates the
    file + header on first call.
    """
    header = ["step", "train/loss", "val/loss", "val/l1"]
    write_header = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([
            int(step),
            "" if train_loss is None else float(train_loss),
            "" if val_loss is None else float(val_loss),
            "" if val_l1 is None else float(val_l1),
        ])


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
    # Local train/val loss log, same schema ACT's CSVLogger writes, so the
    # eval plotting pipeline (build_all_checkpoints_json.py) can read GCBC
    # losses directly from <save_dir>/logs/metrics.csv.
    metrics_csv_path = os.path.join(args.save_dir, "logs", "metrics.csv")

    # Ablation mode resolution
    _use_ablation = False
    _ablation_mode = None
    _use_image = True
    _needs_proprio = args.use_proprio
    _effective_proprio_dim = None  # resolved after example batch

    _base_keys = parse_base_proprio_keys(args.base_proprio_keys)

    if args.proprio_ablation_mode is not None:
        _use_ablation = True
        _ablation_mode = args.proprio_ablation_mode
        if args.policy != "gcbc":
            raise ValueError(
                "--proprio_ablation_mode is only wired in the training loop for "
                f"--policy gcbc (GCBCPolicy + apply_ablation_to_proprio). Got --policy {args.policy}."
            )
        _use_image = ablation_uses_image(_ablation_mode)
        _needs_proprio = ablation_uses_proprio(_ablation_mode)
        _effective_proprio_dim = ablation_proprio_dim(_ablation_mode, _base_keys)

    _train_rng = torch.Generator(device=device)
    _train_rng.manual_seed(args.seed)

    # Save train config for eval_ispatialgym_batched.py and reproducibility
    _policy_type_map = {"gcbc": "gcbc_torch", "gc_ddpm_bc": "gcbc_torch", "gc_iql": "gcbc_torch"}
    train_config = {
        "policy": _policy_type_map.get(args.policy, "gcbc_torch"),
        "use_proprio": args.use_proprio,
        "add_eef_proprio": args.add_eef_proprio,
        "normalize_proprio": args.normalize_proprio,
        "proprio_ablation_mode": args.proprio_ablation_mode,
        "base_proprio_keys": args.base_proprio_keys,
        "proprio_noise_std": args.proprio_noise_std,
        # Training hyperparams for reproducibility
        "policy_arch": args.policy,
        "encoder": args.encoder,
        "encoder_model_name_or_path": args.encoder_model_name_or_path,
        "train_encoder": args.train_encoder,
        "obs_horizon": args.obs_horizon,
        "obs_history_stride": args.obs_history_stride,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "num_steps": args.num_steps,
        "num_epochs": args.num_epochs,
        "seed": args.seed,
        "augment": args.augment,
        "run_name": args.run_name,
        "tfrecord_dir": args.tfrecord_dir,
        "save_dir": args.save_dir,
        # Policy-specific hyperparams
        "diffusion_steps": args.diffusion_steps,
        "target_update_rate": args.target_update_rate,
        "discount": args.discount,
        "expectile": args.expectile,
        "temperature": args.temperature,
        "negative_proportion": args.negative_proportion,
    }
    config_path = os.path.join(args.save_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(train_config, f, indent=4)
    print(f"Saved train config to {config_path}")

    # Detect image encoding from metadata
    metadata_path = os.path.join(args.tfrecord_dir, "action_proprio_metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    image_encoding = metadata.get("image_encoding", "jpeg")

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
        augment=args.augment, augment_kwargs=augment_kwargs,
        use_proprio=args.use_proprio, add_eef_proprio=args.add_eef_proprio,
        normalize_proprio=args.normalize_proprio,
        image_encoding=image_encoding,
        force_full_proprio=_use_ablation,
        obs_horizon=args.obs_horizon,
        obs_history_stride=args.obs_history_stride,
    )

    val_dataset = build_tf_dataset(
        val_paths, args.batch_size, args.seed + 1, train=False,
        augment=False, augment_kwargs=None,
        use_proprio=args.use_proprio, add_eef_proprio=args.add_eef_proprio,
        normalize_proprio=args.normalize_proprio,
        image_encoding=image_encoding,
        force_full_proprio=_use_ablation,
        obs_horizon=args.obs_horizon,
        obs_history_stride=args.obs_history_stride,
    )

    # Load vis trajectories
    vis_trajs = load_raw_trajectories(val_paths, n=3, seed=args.seed,
                                      image_encoding=image_encoding)
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
            augment=args.augment, augment_kwargs=augment_kwargs,
            use_proprio=args.use_proprio, add_eef_proprio=args.add_eef_proprio,
            normalize_proprio=args.normalize_proprio,
            image_encoding=image_encoding,
            force_full_proprio=_use_ablation,
            obs_horizon=args.obs_horizon,
            obs_history_stride=args.obs_history_stride,
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

    if _use_ablation:
        proprio_dim = _effective_proprio_dim
        print(f"Ablation mode: {_ablation_mode} | use_image={_use_image} "
              f"use_proprio={_needs_proprio} proprio_dim={proprio_dim}")
    elif args.use_proprio:
        proprio_dim = example_batch["obs_proprio"].shape[-1]
        print(f"Proprio dim: {proprio_dim} (add_eef={args.add_eef_proprio}, "
              f"normalize={args.normalize_proprio})")
    else:
        proprio_dim = 23

    # Discover checkpoint before model construction (fail fast on bad --resume)
    ckpt = None
    if args.resume:
        ckpt_files = sorted(glob.glob(os.path.join(args.save_dir, "checkpoint_*.pt")))
        if not ckpt_files:
            raise FileNotFoundError(
                f"--resume requested but no checkpoints found in {args.save_dir}")
        ckpt_path = max(ckpt_files,
                        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract encoder config from checkpoint for architecture reconstruction
    encoder_config_dict = ckpt.get("encoder_config") if ckpt is not None else None

    # Create model
    _model_use_proprio = _needs_proprio if _use_ablation else args.use_proprio
    _model_proprio_dim = proprio_dim if _model_use_proprio else 23

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
            use_proprio=_model_use_proprio,
            proprio_dim=_model_proprio_dim,
            hidden_dims=(256, 256, 256),
            dropout_rate=0.1,
            encoder=args.encoder,
            encoder_model_name_or_path=args.encoder_model_name_or_path,
            train_encoder=args.train_encoder,
            load_pretrained_weights=not args.resume,
            encoder_config_dict=encoder_config_dict,
            use_image=_use_image,
            obs_horizon=args.obs_horizon,
            obs_horizon_stride=args.obs_history_stride,
        ).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Policy: {args.policy}, Encoder: {args.encoder}")
    if args.encoder != "resnetv1-34-bridge":
        encoder_id = args.encoder_model_name_or_path or "(default HF)"
        print(f"  Encoder model: {encoder_id}")
        print(f"  Encoder frozen: {not args.train_encoder}")
    print(f"  Total parameters: {n_total:,}")
    print(f"  Trainable parameters: {n_trainable:,}")

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
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = get_lr_schedule(optimizer, args.warmup_steps, total_steps, args.lr)

    # Resume from checkpoint
    start_step = 0
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if target_state_dict is not None and "target_state_dict" in ckpt:
            target_state_dict = ckpt["target_state_dict"]
        if target_model is not None and "target_state_dict" in ckpt:
            target_model.load_state_dict(ckpt["target_state_dict"])
        start_step = ckpt["step"]
        print(f"Resumed from {ckpt_path} at step {start_step}")

    # Extract encoder config for saving in checkpoints (enables fully offline resume)
    _encoder_config = None
    if hasattr(model, "pretrained_encoder"):
        _encoder_config = model.pretrained_encoder.backbone.config.to_dict()

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
            if _use_ablation:
                raw_proprio = batch["obs_proprio"].to(device)  # always (B, 37)
                raw_state = batch.get("obs_state256")
                if raw_state is not None:
                    raw_state = raw_state.to(device)
                proprio = apply_ablation_to_proprio(
                    raw_proprio,
                    _ablation_mode,
                    seed=args.seed,
                    rng=_train_rng,
                    raw_state_256=raw_state,
                    base_keys=_base_keys,
                    noise_std=args.proprio_noise_std,
                )
            else:
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
            # Persist the training loss locally for the eval plots.
            _append_metrics_csv(metrics_csv_path, step, train_loss=loss_val)
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
                        if _use_ablation:
                            v_raw_proprio = val_batch["obs_proprio"].to(device)
                            v_raw_state = val_batch.get("obs_state256")
                            if v_raw_state is not None:
                                v_raw_state = v_raw_state.to(device)
                            v_proprio = apply_ablation_to_proprio(
                                v_raw_proprio,
                                _ablation_mode,
                                seed=args.seed,
                                raw_state_256=v_raw_state,
                                base_keys=_base_keys,
                                noise_std=args.proprio_noise_std,
                            )
                        else:
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
                # Persist val loss (primary optimized loss) + mse for the eval
                # plots. mse goes in the ``val/l1`` column as the secondary
                # metric slot (GCBC has no L1; mse is the action regression
                # error). build_all_checkpoints_json.py reads both.
                _val_loss_key = ("ddpm_loss" if args.policy == "gc_ddpm_bc"
                                 else "total_loss" if args.policy == "gc_iql"
                                 else "actor_loss")
                _append_metrics_csv(
                    metrics_csv_path,
                    step,
                    val_loss=val_summary.get(_val_loss_key),
                    val_l1=val_summary.get("mse"),
                )

            # Visualization (skip for ablation modes to avoid complexity)
            if vis_trajs and not _use_ablation:
                visualize_predictions(
                    model, vis_trajs, step=step,
                    save_dir=args.save_dir,
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
            if _encoder_config is not None:
                ckpt_data["encoder_config"] = _encoder_config
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
    if _encoder_config is not None:
        ckpt_data["encoder_config"] = _encoder_config
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
                        help="Directory with train/val TFRecords")
    parser.add_argument("--save_dir", type=str, default="outputs/gcbc_torch")
    parser.add_argument("--run_name", type=str, default="gcbc_torch")
    parser.add_argument("--encoder", type=str, default="resnetv1-34-bridge",
                        choices=["resnetv1-34-bridge", "dinov2-base", "siglip-base",
                                 "dinov3-vitl16"],
                        help="Vision encoder: resnetv1-34-bridge, dinov2-base, "
                             "siglip-base, dinov3-vitl16")
    parser.add_argument("--encoder_model_name_or_path", type=str, default=None,
                        help="HF repo id or local dir for pretrained encoder weights")
    parser.add_argument("--train_encoder", action="store_true",
                        help="Unfreeze pretrained encoder (dinov2/siglip/dinov3 only)")
    parser.add_argument("--obs_horizon", type=int, default=1,
                        help="Number of stacked observation frames (incl. current) "
                             "fed to the policy. 1 = single-frame (original "
                             "behavior, the default baseline); >1 gives the policy "
                             "temporal history (e.g. 8). Currently only supported "
                             "with --policy gcbc.")
    parser.add_argument("--obs_history_stride", type=int, default=1,
                        help="Temporal spacing (in original timesteps) between the "
                             "stacked observation-history frames. 1 = consecutive "
                             "frames (which at the native rate can be only "
                             "milliseconds apart and nearly identical); a larger "
                             "stride spreads the window over a meaningful span "
                             "(e.g. 3 at 30Hz => 100ms between frames, so "
                             "obs_horizon=8 covers ~0.7s). Only meaningful with "
                             "--obs_horizon > 1.")

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
    parser.add_argument("--proprio_ablation_mode", type=str, default=None,
                        choices=list(ABLATION_MODES),
                        help="Proprioception ablation mode. Replaces --use_proprio/--add_eef_proprio.")
    parser.add_argument(
        "--base_proprio_keys",
        type=str,
        default="base_qvel",
        help="Comma-separated base fields for base_only / shuffled_base_only / state_only_base. "
             "Options: base_qvel, base_qpos, robot_2d_ori, robot_pos (see gcbc_torch/proprio.py).",
    )
    parser.add_argument(
        "--proprio_noise_std",
        type=float,
        default=1.0,
        help="Std scale for random_noise_full_proprio_same_arch (Gaussian, default 1.0).",
    )

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
    if args.proprio_ablation_mode is not None:
        parse_base_proprio_keys(args.base_proprio_keys)  # validate early
    if args.encoder != "resnetv1-34-bridge" and args.policy != "gcbc":
        raise ValueError(
            f"Pretrained encoder '{args.encoder}' is only supported with "
            f"--policy gcbc. Got --policy {args.policy}."
        )
    if args.obs_horizon < 1:
        raise ValueError(f"--obs_horizon must be >= 1, got {args.obs_horizon}.")
    if args.obs_horizon > 1 and args.policy != "gcbc":
        raise ValueError(
            "--obs_horizon > 1 (observation history) is currently only wired "
            f"for --policy gcbc (GCBCPolicy). Got --policy {args.policy}."
        )
    if args.obs_horizon > 1 and args.proprio_ablation_mode is not None:
        raise ValueError(
            "--obs_horizon > 1 is not supported together with "
            "--proprio_ablation_mode (the ablation transforms expect a single "
            "proprio frame). Use obs_horizon=1 for ablation studies."
        )
    if args.obs_history_stride < 1:
        raise ValueError(
            f"--obs_history_stride must be >= 1, got {args.obs_history_stride}."
        )
    if args.obs_history_stride > 1 and args.obs_horizon == 1:
        print(
            "WARNING: --obs_history_stride > 1 has no effect when "
            "--obs_horizon == 1 (single frame). Ignoring stride."
        )
    train(args)


if __name__ == "__main__":
    main()
