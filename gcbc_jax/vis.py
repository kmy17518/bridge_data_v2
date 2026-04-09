"""Validation visualization: side-by-side obs/goal videos + action plots.

At each eval interval, runs inference on 3 fixed validation trajectories,
creates GIF videos (obs | goal) and action comparison plots (pred vs GT),
and logs them to WandB + saves locally.
"""

import os

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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


def load_vis_trajectories(val_tfrecord_paths, n=3, seed=42):
    """Load n raw trajectories from validation TFRecords for visualization."""
    rng = np.random.RandomState(seed)
    n = min(n, len(val_tfrecord_paths))
    selected = rng.choice(len(val_tfrecord_paths), size=n, replace=False)

    trajectories = []
    for idx in sorted(selected):
        path = val_tfrecord_paths[idx]
        raw_dataset = tf.data.TFRecordDataset(path)
        for raw_record in raw_dataset:
            features = {k: tf.io.FixedLenFeature([], tf.string) for k in [
                "observations/images0", "observations/state",
                "actions", "goal_image",
            ]}
            parsed = tf.io.parse_single_example(raw_record, features)
            obs_jpegs = tf.io.parse_tensor(
                parsed["observations/images0"], tf.string)
            obs_images = tf.map_fn(
                lambda j: tf.io.decode_jpeg(j, channels=3),
                obs_jpegs, fn_output_signature=tf.uint8).numpy()
            goal_jpeg = tf.io.parse_tensor(
                parsed["goal_image"], tf.string)
            goal_image = tf.io.decode_jpeg(goal_jpeg, channels=3).numpy()
            trajectories.append({
                "obs_images": obs_images,
                "obs_state": tf.io.parse_tensor(
                    parsed["observations/state"], tf.float32).numpy(),
                "actions": tf.io.parse_tensor(
                    parsed["actions"], tf.float32).numpy(),
                "goal_image": goal_image,
                "name": os.path.splitext(os.path.basename(path))[0],
            })
            break  # one trajectory per TFRecord
    return trajectories


def run_trajectory_inference(agent, traj, action_metadata, rng,
                             use_proprio=False, add_eef=False,
                             normalize_proprio=False, chunk_size=64):
    """Run model on a full trajectory. Returns (denormalized_pred_actions, rng)."""
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

    # Process in chunks to avoid OOM on long trajectories
    all_pred = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk_obs = {
            "image": jnp.array(obs_images[start:end]),
            "proprio": jnp.array(proprio[start:end]),
        }
        chunk_goals = {
            "image": jnp.broadcast_to(
                jnp.array(goal_image[np.newaxis]),
                chunk_obs["image"].shape),
            "proprio": jnp.zeros_like(chunk_obs["proprio"]),
        }
        rng, sub_rng = jax.random.split(rng)
        pred = agent.sample_actions(
            chunk_obs, chunk_goals, seed=sub_rng,
            temperature=1.0, argmax=True)
        all_pred.append(np.array(pred))

    pred_norm = np.concatenate(all_pred, axis=0)  # (T, action_dim) normalized

    # Denormalize
    action_mean = np.array(action_metadata["action"]["mean"])
    action_std = np.array(action_metadata["action"]["std"])
    pred_raw = pred_norm * action_std + action_mean

    return pred_raw, rng


def create_vis_frames(obs_images, goal_image):
    """Side-by-side video frames: obs | 4px separator | goal.

    Returns (T, H, 2*W+4, 3) uint8.
    """
    T, H, W, C = obs_images.shape
    sep = np.full((T, H, 4, C), 255, dtype=np.uint8)
    goal_bc = np.broadcast_to(goal_image[np.newaxis], (T, H, W, C)).copy()
    return np.concatenate([obs_images, sep, goal_bc], axis=2)


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

    # Hide unused subplots
    for d in range(action_dim, len(axes)):
        axes[d].set_visible(False)

    mse = float(np.mean((pred_actions - gt_actions) ** 2))
    fig.suptitle(f"Actions — step {step}  (MSE={mse:.4f})", fontsize=12)
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())  # (H, W, 4) uint8
    img = buf[:, :, :3].copy()  # drop alpha
    plt.close(fig)
    return img, mse


def save_gif(frames, path, fps=10):
    """Save (T, H, W, 3) uint8 as GIF."""
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        path, save_all=True, append_images=pil_frames[1:],
        duration=int(1000 / fps), loop=0)


def visualize_predictions(agent, vis_trajs, step, save_dir, action_metadata, rng,
                          use_wandb=False, use_proprio=False, add_eef=False,
                          normalize_proprio=False, fps=10):
    """Create and save visualizations for fixed validation trajectories.

    Saves to {save_dir}/vis/step_{step}/ and optionally logs to WandB.
    Returns updated rng.
    """
    vis_dir = os.path.join(save_dir, "vis", f"step_{step}")
    os.makedirs(vis_dir, exist_ok=True)

    wandb_logs = {}

    for i, traj in enumerate(vis_trajs):
        pred_actions, rng = run_trajectory_inference(
            agent, traj, action_metadata, rng,
            use_proprio=use_proprio, add_eef=add_eef,
            normalize_proprio=normalize_proprio)

        gt_actions = traj["actions"]  # raw (not normalized)

        # Action comparison plot
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
    return rng
