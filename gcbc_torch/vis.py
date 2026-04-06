"""Validation visualization: action plots for fixed val trajectories.

Identical to gcbc_jax/vis.py but using PyTorch model for inference.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .proprio import extract_proprio_np, normalize_proprio_bounds_np

# R1Pro 23-dim action labels
ACTION_LABELS = [
    "base_x", "base_y", "base_yaw",
    "torso_0", "torso_1", "torso_2", "torso_3",
    "L_arm_0", "L_arm_1", "L_arm_2", "L_arm_3",
    "L_arm_4", "L_arm_5", "L_arm_6",
    "L_grip",
    "R_arm_0", "R_arm_1", "R_arm_2", "R_arm_3",
    "R_arm_4", "R_arm_5", "R_arm_6",
    "R_grip",
]


def run_trajectory_inference(model, traj, action_metadata, device,
                             use_proprio=False, add_eef=False,
                             normalize_proprio=False, chunk_size=64,
                             target_state_dict=None):
    """Run model on a full trajectory. Returns denormalized predictions."""
    obs_images = traj["obs_images"]   # (T, H, W, 3) uint8
    obs_state = traj["obs_state"]     # (T, 256) float32
    goal_image = traj["goal_image"]   # (H, W, 3) uint8
    T = obs_images.shape[0]
    action_dim = traj["actions"].shape[-1]

    # Build proprio
    if use_proprio:
        proprio = extract_proprio_np(obs_state, add_eef=add_eef)
        if normalize_proprio:
            proprio = normalize_proprio_bounds_np(proprio, add_eef=add_eef)
    else:
        proprio = np.zeros((T, action_dim), dtype=np.float32)

    # Process in chunks
    all_pred = []
    model.eval()
    with torch.no_grad():
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk_obs = torch.from_numpy(obs_images[start:end]).to(device)
            chunk_goal = torch.from_numpy(
                np.broadcast_to(goal_image[np.newaxis], (end - start, *goal_image.shape))
            ).to(device)
            chunk_proprio = torch.from_numpy(proprio[start:end]).to(device)

            pred = model.get_action(
                chunk_obs, chunk_goal, chunk_proprio, argmax=True,
                target_state_dict=target_state_dict,
            )
            all_pred.append(pred.cpu().numpy())

    pred_norm = np.concatenate(all_pred, axis=0)  # (T, action_dim)

    # Denormalize
    action_mean = np.array(action_metadata["action"]["mean"])
    action_std = np.array(action_metadata["action"]["std"])
    pred_raw = pred_norm * action_std + action_mean

    return pred_raw


def create_action_plot(pred_actions, gt_actions, step):
    """Plot predicted vs GT for all action dims. Returns (H, W, 3) uint8."""
    action_dim = gt_actions.shape[-1]
    ncols = 4
    nrows = (action_dim + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.2 * nrows), sharex=True)
    axes = axes.flatten()

    for d in range(action_dim):
        ax = axes[d]
        ax.plot(gt_actions[:, d], color="blue", alpha=0.7, linewidth=0.8, label="GT")
        ax.plot(pred_actions[:, d], color="red", alpha=0.7, linewidth=0.8,
                linestyle="--", label="pred")
        label = ACTION_LABELS[d] if d < len(ACTION_LABELS) else f"dim_{d}"
        ax.set_title(label, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
        if d == 0:
            ax.legend(fontsize=7, loc="upper right")

    for d in range(action_dim, len(axes)):
        axes[d].set_visible(False)

    mse = float(np.mean((pred_actions - gt_actions) ** 2))
    fig.suptitle(f"Actions -- step {step}  (MSE={mse:.4f})", fontsize=12)
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = buf[:, :, :3].copy()
    plt.close(fig)
    return img, mse


def save_gif(frames, path, fps=10):
    """Save (T, H, W, 3) uint8 as GIF."""
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        path, save_all=True, append_images=pil_frames[1:],
        duration=int(1000 / fps), loop=0)


def visualize_predictions(model, vis_trajs, step, save_dir, action_metadata,
                          device, use_wandb=False, use_proprio=False,
                          add_eef=False, normalize_proprio=False,
                          target_state_dict=None):
    """Create and save visualizations for fixed validation trajectories."""
    vis_dir = os.path.join(save_dir, "vis", f"step_{step}")
    os.makedirs(vis_dir, exist_ok=True)

    wandb_logs = {}

    for i, traj in enumerate(vis_trajs):
        pred_actions = run_trajectory_inference(
            model, traj, action_metadata, device,
            use_proprio=use_proprio, add_eef=add_eef,
            normalize_proprio=normalize_proprio,
            target_state_dict=target_state_dict)

        gt_actions = traj["actions"]

        action_img, mse = create_action_plot(pred_actions, gt_actions, step)
        plot_path = os.path.join(vis_dir, f"{traj['name']}_actions.png")
        Image.fromarray(action_img).save(plot_path)

        if use_wandb:
            import wandb
            wandb_logs[f"vis/ep{i}_actions"] = wandb.Image(action_img)
            wandb_logs[f"vis/ep{i}_mse"] = mse

    if use_wandb and wandb_logs:
        import wandb
        wandb.log(wandb_logs, step=step)

    print(f"  Vis saved to {vis_dir}", flush=True)
