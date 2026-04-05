"""Proprioception extraction and normalization for R1Pro robot.

Extracts 23-dim or 37-dim proprio from the 256-dim observation.state,
matching the representations used by OpenPI (23-dim) and IL_LIB (37-dim).

Index references: PROPRIOCEPTION_INDICES["R1Pro"] from
    OmniGibson/omnigibson/learning/utils/eval_utils.py

23-dim layout (OpenPI / OpenVLA-OFT):
    [0:3]   base_qvel         -- observation.state[253:256]
    [3:7]   trunk_qpos        -- observation.state[236:240]
    [7:14]  arm_left_qpos     -- observation.state[158:165]
    [14:21] arm_right_qpos    -- observation.state[197:204]
    [21]    gripper_left_sum  -- sum(observation.state[193:195])
    [22]    gripper_right_sum -- sum(observation.state[232:234])

37-dim layout (IL_LIB, extends 23-dim with EEF):
    [23:26] eef_left_pos      -- observation.state[186:189]
    [26:30] eef_left_quat     -- observation.state[189:193]
    [30:33] eef_right_pos     -- observation.state[225:228]
    [33:37] eef_right_quat    -- observation.state[228:232]
"""

import numpy as np
import torch


# === Bounds from JOINT_RANGE["R1Pro"] and EEF_POSITION_RANGE["R1Pro"] ===
# Used for bounds normalization to [-1, 1].

JOINT_BOUNDS_LOW_23 = np.array([
    # base_qvel (3)
    -0.75, -0.75, -1.0,
    # trunk_qpos (4)
    -1.1345, -2.7925, -1.8326, -3.0543,
    # left_arm (7)
    -4.4506, -0.1745, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708,
    # right_arm (7)
    -4.4506, -3.1416, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708,
    # left_gripper_sum (1) -- 2 fingers each [0, 0.05]
    0.0,
    # right_gripper_sum (1)
    0.0,
], dtype=np.float32)

JOINT_BOUNDS_HIGH_23 = np.array([
    # base_qvel (3)
    0.75, 0.75, 1.0,
    # trunk_qpos (4)
    1.8326, 2.5307, 1.5708, 3.0543,
    # left_arm (7)
    1.3090, 3.1416, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708,
    # right_arm (7)
    1.3090, 0.1745, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708,
    # left_gripper_sum (1) -- sum of 2 x [0, 0.05]
    0.1,
    # right_gripper_sum (1)
    0.1,
], dtype=np.float32)

# EEF bounds: positions from EEF_POSITION_RANGE, quaternions identity [-1, 1]
EEF_BOUNDS_LOW = np.array([
    # eef_left_pos (3)
    0.0, -0.65, 0.0,
    # eef_left_quat (4) -- quaternions not normalized, identity bounds
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


# -- NumPy versions (for eval / conversion) ----------------------------------------

def extract_proprio_np(state_256, add_eef=False):
    """Extract 23 or 37 dim proprio from (..., 256) state array."""
    parts = [
        state_256[..., 253:256],                                        # base_qvel (3)
        state_256[..., 236:240],                                        # trunk_qpos (4)
        state_256[..., 158:165],                                        # arm_left_qpos (7)
        state_256[..., 197:204],                                        # arm_right_qpos (7)
        state_256[..., 193:195].sum(axis=-1, keepdims=True),            # gripper_left (1)
        state_256[..., 232:234].sum(axis=-1, keepdims=True),            # gripper_right (1)
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
    """Normalize proprio to [-1, 1] using joint range bounds."""
    if add_eef:
        low = np.concatenate([JOINT_BOUNDS_LOW_23, EEF_BOUNDS_LOW])
        high = np.concatenate([JOINT_BOUNDS_HIGH_23, EEF_BOUNDS_HIGH])
    else:
        low = JOINT_BOUNDS_LOW_23
        high = JOINT_BOUNDS_HIGH_23
    return (2.0 * (proprio - low) / (high - low + 1e-8) - 1.0).astype(np.float32)


# -- TensorFlow versions (for tf.data pipeline) ------------------------------------

def extract_proprio_tf(state_256, add_eef=False):
    """Extract 23 or 37 dim proprio from (..., 256) TF tensor."""
    import tensorflow as tf
    parts = [
        state_256[..., 253:256],
        state_256[..., 236:240],
        state_256[..., 158:165],
        state_256[..., 197:204],
        tf.reduce_sum(state_256[..., 193:195], axis=-1, keepdims=True),
        tf.reduce_sum(state_256[..., 232:234], axis=-1, keepdims=True),
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
    """Normalize proprio to [-1, 1] using joint range bounds."""
    import tensorflow as tf
    if add_eef:
        low = tf.constant(np.concatenate([JOINT_BOUNDS_LOW_23, EEF_BOUNDS_LOW]))
        high = tf.constant(np.concatenate([JOINT_BOUNDS_HIGH_23, EEF_BOUNDS_HIGH]))
    else:
        low = tf.constant(JOINT_BOUNDS_LOW_23)
        high = tf.constant(JOINT_BOUNDS_HIGH_23)
    return 2.0 * (proprio - low) / (high - low + 1e-8) - 1.0


# -- PyTorch versions (for torch inference) ----------------------------------------

def extract_proprio_torch(state_256, add_eef=False):
    """Extract 23 or 37 dim proprio from (..., 256) torch tensor."""
    parts = [
        state_256[..., 253:256],
        state_256[..., 236:240],
        state_256[..., 158:165],
        state_256[..., 197:204],
        state_256[..., 193:195].sum(dim=-1, keepdim=True),
        state_256[..., 232:234].sum(dim=-1, keepdim=True),
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
    """Normalize proprio to [-1, 1] using joint range bounds."""
    if add_eef:
        low = torch.tensor(np.concatenate([JOINT_BOUNDS_LOW_23, EEF_BOUNDS_LOW]),
                           device=proprio.device)
        high = torch.tensor(np.concatenate([JOINT_BOUNDS_HIGH_23, EEF_BOUNDS_HIGH]),
                            device=proprio.device)
    else:
        low = torch.tensor(JOINT_BOUNDS_LOW_23, device=proprio.device)
        high = torch.tensor(JOINT_BOUNDS_HIGH_23, device=proprio.device)
    return 2.0 * (proprio - low) / (high - low + 1e-8) - 1.0
