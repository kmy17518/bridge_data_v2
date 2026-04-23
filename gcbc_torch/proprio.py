"""Proprioception extraction and normalization for R1Pro robot.

Extracts 23-dim or 37-dim proprio from the 256-dim observation.state,
matching the action vector layout (grippers interleaved between arms).

Index references: PROPRIOCEPTION_INDICES["R1Pro"] from
    OmniGibson/omnigibson/learning/utils/eval_utils.py

23-dim layout (matches action vector):
    [0:3]   base_qvel          -- base_link frame; rotated from world-frame
                                  observation.state[253:256] by yaw at
                                  observation.state[246] (so it lives in the
                                  same frame the HolonomicBaseJointController
                                  interprets action[0:3] in)
    [3:7]   trunk_qpos         -- observation.state[236:240]
    [7:14]  arm_left_qpos      -- observation.state[158:165]
    [14]    gripper_left_mean  -- mean(observation.state[193:195])
    [15:22] arm_right_qpos     -- observation.state[197:204]
    [22]    gripper_right_mean -- mean(observation.state[232:234])

37-dim layout (extends 23-dim with EEF):
    [23:26] eef_left_pos       -- observation.state[186:189]
    [26:30] eef_left_quat      -- observation.state[189:193]
    [30:33] eef_right_pos      -- observation.state[225:228]
    [33:37] eef_right_quat     -- observation.state[228:232]
"""

import numpy as np
import torch


# === Bounds from JOINT_RANGE["R1Pro"] and EEF_POSITION_RANGE["R1Pro"] ===
# Used for bounds normalization to [-1, 1].

# Gripper midpoint for binarization: (0.0 + 0.05) / 2
# Each gripper finger is in [0, 0.05]; after mean, the value is in [0, 0.05].
GRIPPER_MIDPOINT = 0.025

# EEF bounds: positions from EEF_POSITION_RANGE (quats are NOT normalized)
EEF_BOUNDS_LOW = np.array([
    # eef_left_pos (3)
    0.0, -0.65, 0.0,
    # eef_left_quat (4) -- raw passthrough, bounds unused
    -1.0, -1.0, -1.0, -1.0,
    # eef_right_pos (3)
    0.0, -0.65, 0.0,
    # eef_right_quat (4)
    -1.0, -1.0, -1.0, -1.0,
], dtype=np.float32)

EEF_BOUNDS_HIGH = np.array([
    # eef_left_pos (3)
    0.65, 0.65, 2.5,
    # eef_left_quat (4)
    1.0, 1.0, 1.0, 1.0,
    # eef_right_pos (3)
    0.65, 0.65, 2.5,
    # eef_right_quat (4)
    1.0, 1.0, 1.0, 1.0,
], dtype=np.float32)

# === Action bounds (ACTION_QPOS_INDICES["R1Pro"] ordering) ===
# Layout: base(3) + torso(4) + left_arm(7) + left_gripper(1) + right_arm(7) + right_gripper(1)
#
# Action-side passthrough indices — these dims skip min-max normalization because
# the recorded action is already in the policy's [-1, 1] target space:
# - Base xy/yaw [0:3]: recorded action is the controller's INPUT (HolonomicBaseJointController
#   scales it by 0.75/0.75/1.0 internally to produce m/s, m/s, rad/s).
# - Grippers [14, 22]: smooth-mode gripper command is binarized in [-1, 1].
# Note: proprio normalization does NOT use this mask — the proprio's base_qvel is in
# physical units (m/s, rad/s), so its natural range really is ACTION_BOUNDS_LOW/HIGH.
ACTION_GRIPPER_INDICES = [14, 22]
ACTION_PASSTHROUGH_INDICES = [0, 1, 2, 14, 22]
ACTION_NORMALIZE_MASK_23 = np.ones(23, dtype=bool)
ACTION_NORMALIZE_MASK_23[ACTION_PASSTHROUGH_INDICES] = False

ACTION_BOUNDS_LOW_23 = np.array([
    # base (3)
    -0.75, -0.75, -1.0,
    # torso (4)
    -1.1345, -2.7925, -1.8326, -3.0543,
    # left_arm (7)
    -4.4506, -0.1745, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708,
    # left_gripper (1) -- pass-through, placeholder
    0.0,
    # right_arm (7)
    -4.4506, -3.1416, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708,
    # right_gripper (1) -- pass-through, placeholder
    0.0,
], dtype=np.float32)

ACTION_BOUNDS_HIGH_23 = np.array([
    # base (3)
    0.75, 0.75, 1.0,
    # torso (4)
    1.8326, 2.5307, 1.5708, 3.0543,
    # left_arm (7)
    1.3090, 3.1416, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708,
    # left_gripper (1) -- placeholder
    1.0,
    # right_arm (7)
    1.3090, 0.1745, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708,
    # right_gripper (1) -- placeholder
    1.0,
], dtype=np.float32)


# -- NumPy versions (for eval / conversion) ----------------------------------------

def extract_proprio_np(state_256, add_eef=False):
    """Extract 23 or 37 dim proprio from (..., 256) state array."""
    yaw = state_256[..., 246]
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    vx_w = state_256[..., 253]
    vy_w = state_256[..., 254]
    base_qvel = np.stack([
        cos_y * vx_w + sin_y * vy_w,
        -sin_y * vx_w + cos_y * vy_w,
        state_256[..., 255],
    ], axis=-1)
    parts = [
        base_qvel,                                                      # base_qvel (3, base frame)
        state_256[..., 236:240],                                        # trunk_qpos (4)
        state_256[..., 158:165],                                        # arm_left_qpos (7)
        state_256[..., 193:195].mean(axis=-1, keepdims=True),           # gripper_left (1)
        state_256[..., 197:204],                                        # arm_right_qpos (7)
        state_256[..., 232:234].mean(axis=-1, keepdims=True),           # gripper_right (1)
    ]
    if add_eef:
        parts.extend([
            state_256[..., 186:189],                                    # eef_left_pos (3)
            state_256[..., 189:193],                                    # eef_left_quat (4)
            state_256[..., 225:228],                                    # eef_right_pos (3)
            state_256[..., 228:232],                                    # eef_right_quat (4)
        ])
    return np.concatenate(parts, axis=-1).astype(np.float32)


def normalize_proprio_bounds_np(proprio, add_eef=False):
    """Normalize proprio to [-1, 1] matching action-layout per-group logic.

    - [0:14]  (base+trunk+left_arm): min-max to [-1, 1]
    - [14]    (gripper_left): binarize at GRIPPER_MIDPOINT to {-1, 1}
    - [15:22] (right_arm): min-max to [-1, 1]
    - [22]    (gripper_right): binarize at GRIPPER_MIDPOINT to {-1, 1}
    - [23:26], [30:33] (EEF pos): min-max to [-1, 1]
    - [26:30], [33:37] (EEF quat): raw passthrough
    """
    result = proprio.copy()
    low = ACTION_BOUNDS_LOW_23
    high = ACTION_BOUNDS_HIGH_23
    result[..., :14] = 2.0 * (result[..., :14] - low[:14]) / (high[:14] - low[:14] + 1e-8) - 1.0
    result[..., 14] = np.where(result[..., 14] > GRIPPER_MIDPOINT, 1.0, -1.0)
    result[..., 15:22] = 2.0 * (result[..., 15:22] - low[15:22]) / (high[15:22] - low[15:22] + 1e-8) - 1.0
    result[..., 22] = np.where(result[..., 22] > GRIPPER_MIDPOINT, 1.0, -1.0)
    if add_eef:
        result[..., 23:26] = 2.0 * (result[..., 23:26] - EEF_BOUNDS_LOW[:3]) / (EEF_BOUNDS_HIGH[:3] - EEF_BOUNDS_LOW[:3] + 1e-8) - 1.0
        # eef_left_quat [26:30]: raw passthrough
        result[..., 30:33] = 2.0 * (result[..., 30:33] - EEF_BOUNDS_LOW[7:10]) / (EEF_BOUNDS_HIGH[7:10] - EEF_BOUNDS_LOW[7:10] + 1e-8) - 1.0
        # eef_right_quat [33:37]: raw passthrough
    return result.astype(np.float32)


# -- TensorFlow versions (for tf.data pipeline) ------------------------------------

def extract_proprio_tf(state_256, add_eef=False):
    """Extract 23 or 37 dim proprio from (..., 256) TF tensor."""
    import tensorflow as tf
    yaw = state_256[..., 246]
    cos_y = tf.cos(yaw)
    sin_y = tf.sin(yaw)
    vx_w = state_256[..., 253]
    vy_w = state_256[..., 254]
    base_qvel = tf.stack([
        cos_y * vx_w + sin_y * vy_w,
        -sin_y * vx_w + cos_y * vy_w,
        state_256[..., 255],
    ], axis=-1)
    parts = [
        base_qvel,                                                      # base_qvel (3, base frame)
        state_256[..., 236:240],                                        # trunk_qpos (4)
        state_256[..., 158:165],                                        # arm_left_qpos (7)
        tf.reduce_mean(state_256[..., 193:195], axis=-1, keepdims=True),# gripper_left (1)
        state_256[..., 197:204],                                        # arm_right_qpos (7)
        tf.reduce_mean(state_256[..., 232:234], axis=-1, keepdims=True),# gripper_right (1)
    ]
    if add_eef:
        parts.extend([
            state_256[..., 186:189],
            state_256[..., 189:193],
            state_256[..., 225:228],
            state_256[..., 228:232],
        ])
    return tf.concat(parts, axis=-1)


def normalize_proprio_bounds_tf(proprio, add_eef=False):
    """Normalize proprio to [-1, 1] matching action-layout per-group logic (TF)."""
    import tensorflow as tf
    low = tf.constant(ACTION_BOUNDS_LOW_23)
    high = tf.constant(ACTION_BOUNDS_HIGH_23)
    # [0:14] base+trunk+left_arm: min-max
    normed_left = 2.0 * (proprio[..., :14] - low[:14]) / (high[:14] - low[:14] + 1e-8) - 1.0
    # [14] gripper_left: binarize
    grip_left = tf.where(proprio[..., 14:15] > GRIPPER_MIDPOINT, 1.0, -1.0)
    # [15:22] right_arm: min-max
    normed_right = 2.0 * (proprio[..., 15:22] - low[15:22]) / (high[15:22] - low[15:22] + 1e-8) - 1.0
    # [22] gripper_right: binarize
    grip_right = tf.where(proprio[..., 22:23] > GRIPPER_MIDPOINT, 1.0, -1.0)
    parts = [normed_left, grip_left, normed_right, grip_right]
    if add_eef:
        eef_l_pos_low = tf.constant(EEF_BOUNDS_LOW[:3])
        eef_l_pos_high = tf.constant(EEF_BOUNDS_HIGH[:3])
        eef_r_pos_low = tf.constant(EEF_BOUNDS_LOW[7:10])
        eef_r_pos_high = tf.constant(EEF_BOUNDS_HIGH[7:10])
        parts.append(2.0 * (proprio[..., 23:26] - eef_l_pos_low) / (eef_l_pos_high - eef_l_pos_low + 1e-8) - 1.0)
        parts.append(proprio[..., 26:30])  # eef_left_quat: passthrough
        parts.append(2.0 * (proprio[..., 30:33] - eef_r_pos_low) / (eef_r_pos_high - eef_r_pos_low + 1e-8) - 1.0)
        parts.append(proprio[..., 33:37])  # eef_right_quat: passthrough
    return tf.concat(parts, axis=-1)


# -- PyTorch versions (for torch inference) ----------------------------------------

def extract_proprio_torch(state_256, add_eef=False):
    """Extract 23 or 37 dim proprio from (..., 256) torch tensor."""
    yaw = state_256[..., 246]
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    vx_w = state_256[..., 253]
    vy_w = state_256[..., 254]
    base_qvel = torch.stack([
        cos_y * vx_w + sin_y * vy_w,
        -sin_y * vx_w + cos_y * vy_w,
        state_256[..., 255],
    ], dim=-1)
    parts = [
        base_qvel,                                                      # base_qvel (3, base frame)
        state_256[..., 236:240],                                        # trunk_qpos (4)
        state_256[..., 158:165],                                        # arm_left_qpos (7)
        state_256[..., 193:195].mean(dim=-1, keepdim=True),             # gripper_left (1)
        state_256[..., 197:204],                                        # arm_right_qpos (7)
        state_256[..., 232:234].mean(dim=-1, keepdim=True),             # gripper_right (1)
    ]
    if add_eef:
        parts.extend([
            state_256[..., 186:189],
            state_256[..., 189:193],
            state_256[..., 225:228],
            state_256[..., 228:232],
        ])
    return torch.cat(parts, dim=-1)


def normalize_proprio_bounds_torch(proprio, add_eef=False):
    """Normalize proprio to [-1, 1] matching action-layout per-group logic (PyTorch)."""
    result = proprio.clone()
    low = torch.tensor(ACTION_BOUNDS_LOW_23, device=proprio.device)
    high = torch.tensor(ACTION_BOUNDS_HIGH_23, device=proprio.device)
    result[..., :14] = 2.0 * (result[..., :14] - low[:14]) / (high[:14] - low[:14] + 1e-8) - 1.0
    result[..., 14] = torch.where(result[..., 14] > GRIPPER_MIDPOINT, 1.0, -1.0)
    result[..., 15:22] = 2.0 * (result[..., 15:22] - low[15:22]) / (high[15:22] - low[15:22] + 1e-8) - 1.0
    result[..., 22] = torch.where(result[..., 22] > GRIPPER_MIDPOINT, 1.0, -1.0)
    if add_eef:
        eef_l_pos_low = torch.tensor(EEF_BOUNDS_LOW[:3], device=proprio.device)
        eef_l_pos_high = torch.tensor(EEF_BOUNDS_HIGH[:3], device=proprio.device)
        eef_r_pos_low = torch.tensor(EEF_BOUNDS_LOW[7:10], device=proprio.device)
        eef_r_pos_high = torch.tensor(EEF_BOUNDS_HIGH[7:10], device=proprio.device)
        result[..., 23:26] = 2.0 * (result[..., 23:26] - eef_l_pos_low) / (eef_l_pos_high - eef_l_pos_low + 1e-8) - 1.0
        # [26:30] eef_left_quat: raw passthrough
        result[..., 30:33] = 2.0 * (result[..., 30:33] - eef_r_pos_low) / (eef_r_pos_high - eef_r_pos_low + 1e-8) - 1.0
        # [33:37] eef_right_quat: raw passthrough
    return result


# -- Action normalization / denormalization ----------------------------------------

def normalize_actions_bounds_tf(actions):
    """Normalize 23-dim actions to [-1, 1] using JOINT_RANGE bounds (TF).

    Trunk + arm dims: min-max normalization to [-1, 1].
    Base + gripper dims [0, 1, 2, 14, 22]: pass through (already in [-1, 1]).
    """
    import tensorflow as tf
    low = tf.constant(ACTION_BOUNDS_LOW_23)
    high = tf.constant(ACTION_BOUNDS_HIGH_23)
    mask = tf.constant(ACTION_NORMALIZE_MASK_23)
    normalized = 2.0 * (actions - low) / (high - low + 1e-8) - 1.0
    return tf.where(mask, normalized, actions)


def normalize_actions_bounds_np(actions):
    """Normalize 23-dim actions to [-1, 1] using JOINT_RANGE bounds (NumPy).

    Trunk + arm dims: min-max normalization to [-1, 1].
    Base dims [0, 1, 2]: pass through (already in [-1, 1]).
    Gripper dims [14, 22]: binarize to {-1, 1}.
    """
    result = actions.copy()
    mask = ACTION_NORMALIZE_MASK_23
    low = ACTION_BOUNDS_LOW_23
    high = ACTION_BOUNDS_HIGH_23
    result[..., mask] = 2.0 * (result[..., mask] - low[mask]) / (high[mask] - low[mask] + 1e-8) - 1.0
    for gi in ACTION_GRIPPER_INDICES:
        result[..., gi] = np.where(result[..., gi] > 0, 1.0, -1.0)
    return result.astype(np.float32)


def denormalize_actions_bounds_np(actions):
    """Denormalize 23-dim actions from [-1, 1] to original range (NumPy).

    Trunk + arm dims: inverse min-max.
    Base dims [0, 1, 2]: pass through (already in env.step input space).
    Gripper dims [14, 22]: binarize to {-1, 1}.
    """
    result = actions.copy()
    mask = ACTION_NORMALIZE_MASK_23
    low = ACTION_BOUNDS_LOW_23
    high = ACTION_BOUNDS_HIGH_23
    result[..., mask] = (result[..., mask] + 1.0) / 2.0 * (high[mask] - low[mask]) + low[mask]
    for gi in ACTION_GRIPPER_INDICES:
        result[..., gi] = np.where(result[..., gi] > 0, 1.0, -1.0)
    return result.astype(np.float32)


def denormalize_actions_bounds_torch(actions):
    """Denormalize 23-dim actions from [-1, 1] to original range (PyTorch).

    Trunk + arm dims: inverse min-max.
    Base dims [0, 1, 2]: pass through (already in env.step input space).
    Gripper dims [14, 22]: binarize to {-1, 1}.
    """
    result = actions.clone()
    mask = torch.tensor(ACTION_NORMALIZE_MASK_23, device=actions.device)
    low = torch.tensor(ACTION_BOUNDS_LOW_23, device=actions.device)
    high = torch.tensor(ACTION_BOUNDS_HIGH_23, device=actions.device)
    denorm = (actions + 1.0) / 2.0 * (high - low) + low
    result = torch.where(mask, denorm, result)
    for gi in ACTION_GRIPPER_INDICES:
        result[..., gi] = torch.where(result[..., gi] > 0, 1.0, -1.0)
    return result
