"""Test that vis.py (training-time) and eval_policy.py produce identical actions
when given the exact same input images, state, and model weights.

Usage:
    cd il/bridge_data_v2
    python -m gcbc_torch.test_path_equivalence \
        --checkpoint_dir outputs/gcbc_torch_task0053_proprio \
        --tfrecord_dir gcbc_jax/tfrecords/task-0053-final
"""

import argparse
import glob
import os
import sys

import numpy as np
import tensorflow as tf
import torch

# Prevent TF from grabbing GPU
tf.config.set_visible_devices([], "GPU")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--tfrecord_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cpu")

    # --- Load checkpoint args to get proprio settings ---
    ckpt_files = sorted(glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_*.pt")))
    assert ckpt_files, f"No checkpoints in {args.checkpoint_dir}"
    ckpt_path = max(ckpt_files,
                    key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    use_proprio = train_args.get("use_proprio", False)
    add_eef = train_args.get("add_eef_proprio", False)
    normalize_proprio = train_args.get("normalize_proprio", False)
    print(f"Checkpoint: {ckpt_path}")
    print(f"  use_proprio={use_proprio}, add_eef={add_eef}, normalize_proprio={normalize_proprio}")

    from .proprio import ACTION_BOUNDS_LOW_23, denormalize_actions_bounds_np
    action_dim = len(ACTION_BOUNDS_LOW_23)  # 23 for R1Pro

    # --- Load one val trajectory (same as vis.py: load_raw_trajectories) ---
    from .dataset import load_raw_trajectories
    val_paths = sorted(glob.glob(os.path.join(args.tfrecord_dir, "val", "*.tfrecord")))
    assert val_paths, f"No val TFRecords in {args.tfrecord_dir}/val/"
    vis_trajs = load_raw_trajectories(val_paths, n=1, seed=args.seed)
    traj = vis_trajs[0]
    print(f"\nTrajectory: {traj['name']}")
    print(f"  obs_images: {traj['obs_images'].shape}, dtype={traj['obs_images'].dtype}")
    print(f"  obs_state:  {traj['obs_state'].shape}")
    print(f"  goal_image: {traj['goal_image'].shape}, dtype={traj['goal_image'].dtype}")
    print(f"  actions:    {traj['actions'].shape}")

    # Use only the first timestep
    obs_image_np = traj["obs_images"][0]    # (H, W, 3) uint8
    goal_image_np = traj["goal_image"]       # (H, W, 3) uint8
    obs_state_np = traj["obs_state"][0]      # (256,) float32
    gt_action = traj["actions"][0]           # (action_dim,) float32

    # --- Build model (shared by both paths) ---
    from .model import GCBCPolicy
    proprio_dim = 37 if add_eef else 23
    model = GCBCPolicy(
        action_dim=action_dim,
        use_proprio=use_proprio,
        proprio_dim=proprio_dim if use_proprio else 23,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # =====================================================================
    # PATH A: vis.py (run_trajectory_inference, single-step)
    # =====================================================================
    from .proprio import extract_proprio_np, normalize_proprio_bounds_np

    if use_proprio:
        proprio_a = extract_proprio_np(obs_state_np, add_eef=add_eef)
        if normalize_proprio:
            proprio_a = normalize_proprio_bounds_np(proprio_a, add_eef=add_eef)
    else:
        proprio_a = np.zeros(action_dim, dtype=np.float32)

    obs_t_a = torch.from_numpy(obs_image_np[np.newaxis]).to(device)
    goal_t_a = torch.from_numpy(goal_image_np[np.newaxis]).to(device)
    proprio_t_a = torch.from_numpy(proprio_a[np.newaxis]).to(device)

    with torch.no_grad():
        pred_norm_a = model.get_action(obs_t_a, goal_t_a, proprio_t_a, argmax=True)
    pred_norm_a = pred_norm_a[0].cpu().numpy()
    action_a = denormalize_actions_bounds_np(pred_norm_a)

    # =====================================================================
    # PATH B: eval_policy.py (TorchGCBCEvalPolicy.forward, simulated)
    # =====================================================================
    # Simulate exactly what eval_policy.forward() does:
    #   1. obs_image from obs dict → numpy uint8, strip alpha
    #   2. Resize to match goal if needed
    #   3. goal from self.goal_image (loaded from PNG)
    #   4. proprio from obs["robot_r1::proprio"]

    # Step 1: simulate obs dict image (same pixels, but go through the same code path)
    head_rgb = torch.from_numpy(obs_image_np)  # simulate obs[head_key] as torch tensor
    obs_image_b = head_rgb.cpu().numpy().astype(np.uint8)
    if obs_image_b.shape[-1] == 4:
        obs_image_b = obs_image_b[..., :3]

    # Step 2: resize check (same as eval_policy.py lines 88-93)
    from PIL import Image
    goal_image_b = goal_image_np  # simulate self.goal_image (loaded from PNG)
    goal_H, goal_W = goal_image_b.shape[:2]
    obs_H, obs_W = obs_image_b.shape[:2]
    if obs_H != goal_H or obs_W != goal_W:
        obs_image_b = np.array(
            Image.fromarray(obs_image_b).resize((goal_W, goal_H))
        )

    # Step 3: build tensors
    obs_t_b = torch.from_numpy(obs_image_b[np.newaxis]).to(device)
    goal_t_b = torch.from_numpy(goal_image_b[np.newaxis]).to(device)

    # Step 4: proprio (simulate obs["robot_r1::proprio"] = 256-dim state)
    if use_proprio:
        proprio_256_b = obs_state_np.astype(np.float32)  # same 256-dim
        proprio_b = extract_proprio_np(proprio_256_b, add_eef=add_eef)
        if normalize_proprio:
            proprio_b = normalize_proprio_bounds_np(proprio_b, add_eef=add_eef)
        proprio_t_b = torch.from_numpy(proprio_b[np.newaxis]).to(device)
    else:
        proprio_t_b = None

    # Step 5: inference (same as eval_policy.py)
    with torch.no_grad():
        actions_b = model.get_action(obs_t_b, goal_t_b, proprio_t_b, argmax=True)

    # Step 6: denormalize
    actions_np_b = actions_b[0].cpu().numpy()
    action_b = denormalize_actions_bounds_np(actions_np_b)

    # =====================================================================
    # COMPARE
    # =====================================================================
    print("\n" + "=" * 60)
    print("COMPARISON: Path A (vis.py) vs Path B (eval_policy.py)")
    print("=" * 60)

    # Check intermediate values
    print(f"\n--- Inputs ---")
    print(f"obs_image identical:  {np.array_equal(obs_image_np, obs_image_b)}")
    print(f"goal_image identical: {np.array_equal(goal_image_np, goal_image_b)}")
    if use_proprio:
        print(f"proprio identical:    {np.array_equal(proprio_a, proprio_b)}")
        print(f"  proprio_a[:5] = {proprio_a[:5]}")
        print(f"  proprio_b[:5] = {proprio_b[:5]}")
    else:
        print(f"proprio: Path A = zeros({proprio_a.shape}), Path B = None")

    print(f"\n--- Normalized predictions (model output) ---")
    print(f"pred_norm_a[:5] = {pred_norm_a[:5]}")
    print(f"pred_norm_b[:5] = {actions_np_b[:5]}")
    norm_diff = np.abs(pred_norm_a - actions_np_b)
    print(f"max abs diff:     {norm_diff.max():.2e}")
    print(f"mean abs diff:    {norm_diff.mean():.2e}")

    print(f"\n--- Denormalized actions (final output) ---")
    print(f"action_a[:5] = {action_a[:5]}")
    print(f"action_b[:5] = {action_b[:5]}")
    action_diff = np.abs(action_a - action_b)
    print(f"max abs diff:     {action_diff.max():.2e}")
    print(f"mean abs diff:    {action_diff.mean():.2e}")

    print(f"\n--- Ground-truth action (from TFRecord, raw) ---")
    print(f"gt_action[:5]  = {gt_action[:5]}")

    # Verdict
    ATOL = 1e-5
    if np.allclose(action_a, action_b, atol=ATOL):
        print(f"\n✓ PASS: Both paths produce identical actions (atol={ATOL})")
    else:
        print(f"\n✗ FAIL: Actions differ beyond atol={ATOL}")
        print(f"  Per-dim diffs:")
        labels = [
            "base_x", "base_y", "base_yaw",
            "torso_0", "torso_1", "torso_2", "torso_3",
            "L_arm_0", "L_arm_1", "L_arm_2", "L_arm_3",
            "L_arm_4", "L_arm_5", "L_arm_6",
            "L_grip",
            "R_arm_0", "R_arm_1", "R_arm_2", "R_arm_3",
            "R_arm_4", "R_arm_5", "R_arm_6",
            "R_grip",
        ]
        for d in range(len(action_a)):
            label = labels[d] if d < len(labels) else f"dim_{d}"
            print(f"    {label:12s}: a={action_a[d]:+.6f}  b={action_b[d]:+.6f}  "
                  f"diff={action_diff[d]:.2e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
