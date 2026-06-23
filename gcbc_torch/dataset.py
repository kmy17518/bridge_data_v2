"""PyTorch dataset that reads TFRecords with fixed goal images.

Uses TensorFlow only for TFRecord parsing, then yields PyTorch tensors.
Applies identical augmentations to the JAX version using torchvision.
"""

import glob
import os

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import IterableDataset
import torchvision.transforms.v2.functional as TF

from .proprio import extract_proprio_tf, normalize_proprio_bounds_tf, normalize_actions_bounds_tf


# TFRecord field types (same as jaxrl_m BridgeDataset + goal_image).
# Image fields store JPEG-compressed string tensors (default) or raw uint8.
IMAGE_KEYS = {"observations/images0", "next_observations/images0", "goal_image"}


def _proto_type_spec(image_encoding="jpeg"):
    """Return PROTO_TYPE_SPEC dict with image dtype matching the encoding."""
    img_dtype = tf.string if image_encoding == "jpeg" else tf.uint8
    return {
        "observations/images0": img_dtype,
        "observations/state": tf.float32,
        "next_observations/images0": img_dtype,
        "next_observations/state": tf.float32,
        "actions": tf.float32,
        "terminals": tf.bool,
        "truncates": tf.bool,
        "goal_image": img_dtype,
    }


PROTO_TYPE_SPEC = _proto_type_spec("jpeg")


def _decode_jpeg_frames(jpeg_tensor):
    """Decode a 1-D string tensor of JPEG bytes into (T, H, W, 3) uint8."""
    return tf.map_fn(
        lambda j: tf.io.decode_jpeg(j, channels=3),
        jpeg_tensor,
        fn_output_signature=tf.uint8,
    )


def _decode_example(example_proto, use_proprio=False, add_eef_proprio=False,
                    image_encoding="jpeg", force_full_proprio=False):
    """Decode a single TFRecord example."""
    proto_spec = _proto_type_spec(image_encoding)
    features = {key: tf.io.FixedLenFeature([], tf.string) for key in proto_spec}
    parsed = tf.io.parse_single_example(example_proto, features)
    tensors = {
        key: tf.io.parse_tensor(parsed[key], dtype)
        for key, dtype in proto_spec.items()
    }

    if image_encoding == "jpeg":
        obs_images = _decode_jpeg_frames(tensors["observations/images0"])
        next_obs_images = _decode_jpeg_frames(tensors["next_observations/images0"])
        goal_image = tf.io.decode_jpeg(tensors["goal_image"], channels=3)
    else:
        obs_images = tensors["observations/images0"]
        next_obs_images = tensors["next_observations/images0"]
        goal_image = tensors["goal_image"]

    obs_proprio = tensors["observations/state"]
    next_obs_proprio = tensors["next_observations/state"]

    obs_state256 = None
    next_state256 = None
    if force_full_proprio:
        obs_state256 = tf.cast(tensors["observations/state"], tf.float32)
        next_state256 = tf.cast(tensors["next_observations/state"], tf.float32)
        obs_proprio = extract_proprio_tf(obs_state256, add_eef=True)
        next_obs_proprio = extract_proprio_tf(next_state256, add_eef=True)
    elif use_proprio:
        obs_proprio = extract_proprio_tf(
            tf.cast(obs_proprio, tf.float32), add_eef=add_eef_proprio)
        next_obs_proprio = extract_proprio_tf(
            tf.cast(next_obs_proprio, tf.float32), add_eef=add_eef_proprio)

    out = {
        "observations": {
            "image": obs_images,
            "proprio": obs_proprio,
        },
        "next_observations": {
            "image": next_obs_images,
            "proprio": next_obs_proprio,
        },
        "actions": tensors["actions"],
        "terminals": tensors["terminals"],
        "truncates": tensors["truncates"],
        "goal_image": goal_image,
    }
    if obs_state256 is not None:
        out["observations"]["state256"] = obs_state256
        out["next_observations"]["state256"] = next_state256
    return out


def _add_goals(traj):
    """Use fixed goal image for all transitions (same as JAX version)."""
    traj_len = tf.shape(traj["actions"])[0]

    goal_image = tf.broadcast_to(
        traj["goal_image"][tf.newaxis],
        [traj_len, tf.shape(traj["goal_image"])[0],
         tf.shape(traj["goal_image"])[1], 3]
    )

    traj["goals"] = {
        "image": goal_image,
        "proprio": traj["observations"]["proprio"],  # dummy
    }
    traj["rewards"] = tf.cast(tf.fill([traj_len], -1), tf.int32)
    traj["masks"] = tf.logical_not(traj["terminals"])
    del traj["goal_image"]
    return traj


def _add_obs_history(traj, obs_horizon, obs_history_stride=1):
    """Stack the past ``obs_horizon`` frames (incl. current) per timestep.

    Operates on an intact trajectory (before ``unbatch``), since the post-
    unbatch shuffle destroys temporal adjacency. For each timestep ``t`` we
    gather observation frames ``[t-(obs_horizon-1)*stride, ..., t-stride, t]``
    (oldest first), clamping negative indices to 0 so episode starts repeat the
    first frame.

    ``obs_history_stride`` controls the temporal *spacing* between stacked
    frames (in original timesteps). At the native data rate, consecutive
    frames (stride 1) can be only milliseconds apart and nearly identical;
    a larger stride spreads the window over a behaviorally meaningful span
    (e.g. stride 3 at 30 Hz -> 100 ms between frames). The window covers
    ``(obs_horizon - 1) * stride + 1`` timesteps.

    Shapes:
        observations/image:   (T, H, W, 3) -> (T, obs_horizon, H, W, 3)
        observations/proprio: (T, P)        -> (T, obs_horizon, P)

    Only ``observations`` is windowed; ``goals`` and ``next_observations``
    are left single-frame (the goal is fixed, and history is an obs-side
    concept for the GCBC policy).
    """
    T = tf.shape(traj["observations"]["image"])[0]
    # (T, obs_horizon) index matrix of clamped, stride-spaced past indices.
    offsets = tf.range(
        -(obs_horizon - 1) * obs_history_stride, 1, obs_history_stride
    )  # [-(H-1)*s, ..., -s, 0]
    idx = tf.range(T)[:, tf.newaxis] + offsets[tf.newaxis, :]
    idx = tf.maximum(idx, 0)

    traj["observations"]["image"] = tf.gather(
        traj["observations"]["image"], idx, axis=0)
    traj["observations"]["proprio"] = tf.gather(
        traj["observations"]["proprio"], idx, axis=0)
    return traj


def _normalize_actions(traj):
    """Normalize actions to [-1, 1] using JOINT_RANGE bounds.

    Trunk + arm dims: min-max normalization to [-1, 1].
    Base + gripper dims [0, 1, 2, 14, 22]: pass through (already in [-1, 1]).
    """
    traj["actions"] = normalize_actions_bounds_tf(traj["actions"])
    return traj


def _normalize_proprio(traj, add_eef=False):
    """Normalize proprio to [-1, 1] using bounds."""
    for key in ["observations", "next_observations"]:
        traj[key]["proprio"] = normalize_proprio_bounds_tf(
            traj[key]["proprio"], add_eef=add_eef)
    return traj


def _augment_images(seed, traj, augment_kwargs):
    """Apply augmentations to obs, next_obs, and goal images using same seed.

    Matches JAX augmentation pipeline exactly:
    - random_resized_crop(scale=[0.8,1.0], ratio=[0.9,1.1])
    - random_brightness(0.2)
    - random_contrast(0.8, 1.2)
    - random_saturation(0.8, 1.2)
    - random_hue(0.1)
    """
    from jaxrl_m.data.tf_augmentations import augment as tf_augment

    sub_seed = [seed, seed]
    for key in ["observations", "next_observations", "goals"]:
        traj[key]["image"] = tf_augment(
            traj[key]["image"], sub_seed, **augment_kwargs)
    return traj


def build_tf_dataset(tfrecord_paths, batch_size, seed, train=True,
                     augment=True, augment_kwargs=None,
                     shuffle_buffer_size=25000,
                     use_proprio=False, add_eef_proprio=False,
                     normalize_proprio=False,
                     image_encoding="jpeg",
                     force_full_proprio=False,
                     obs_horizon=1, obs_history_stride=1):
    """Build a tf.data.Dataset that yields batches of transitions.

    Returns batches as numpy arrays (to be converted to PyTorch tensors).

    When ``obs_horizon > 1``, each transition's ``obs_image`` / ``obs_proprio``
    is a stack of the latest ``obs_horizon`` frames (see ``_add_obs_history``),
    yielding shapes ``(B, obs_horizon, H, W, 3)`` / ``(B, obs_horizon, P)``.
    ``obs_history_stride`` spaces those frames out in original timesteps so the
    window spans a meaningful duration rather than near-identical frames.
    """
    from functools import partial

    # Shuffle file list
    dataset = tf.data.Dataset.from_tensor_slices(tfrecord_paths)
    if train:
        dataset = dataset.shuffle(len(tfrecord_paths), seed)

    # Read TFRecords
    dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)

    # Decode examples
    decode_fn = partial(_decode_example, use_proprio=use_proprio,
                        add_eef_proprio=add_eef_proprio,
                        image_encoding=image_encoding,
                        force_full_proprio=force_full_proprio)
    dataset = dataset.map(decode_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Normalize actions (always, using JOINT_RANGE bounds)
    dataset = dataset.map(
        lambda t: _normalize_actions(t),
        num_parallel_calls=tf.data.AUTOTUNE)

    # Normalize proprio
    _norm_add_eef = True if force_full_proprio else add_eef_proprio
    if (force_full_proprio or (use_proprio and normalize_proprio)):
        dataset = dataset.map(
            lambda t: _normalize_proprio(t, add_eef=_norm_add_eef),
            num_parallel_calls=tf.data.AUTOTUNE)

    # Add goals
    dataset = dataset.map(_add_goals, num_parallel_calls=tf.data.AUTOTUNE)

    # Stack observation history (before unbatch, while trajectories are intact)
    if obs_horizon > 1:
        dataset = dataset.map(
            lambda t: _add_obs_history(t, obs_horizon, obs_history_stride),
            num_parallel_calls=tf.data.AUTOTUNE)

    # Unbatch trajectories into transitions
    dataset = dataset.unbatch()

    if train:
        dataset = dataset.shuffle(shuffle_buffer_size, seed).repeat()

    # Augmentation
    if train and augment and augment_kwargs is not None:
        dataset = dataset.enumerate(start=seed)
        dataset = dataset.map(
            lambda s, t: _augment_images(s, t, augment_kwargs),
            num_parallel_calls=tf.data.AUTOTUNE)

    # Batch
    dataset = dataset.batch(batch_size, drop_remainder=True,
                            num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(1)

    return dataset


def tf_batch_to_torch(batch, device="cpu"):
    """Convert a TF batch (numpy) to PyTorch tensors."""
    d = {
        "obs_image": torch.from_numpy(batch["observations"]["image"]),
        "goal_image": torch.from_numpy(batch["goals"]["image"]),
        "obs_proprio": torch.from_numpy(batch["observations"]["proprio"]),
        "actions": torch.from_numpy(batch["actions"]),
    }
    if "state256" in batch["observations"]:
        d["obs_state256"] = torch.from_numpy(batch["observations"]["state256"])
    return d


def tf_batch_to_torch_iql(batch, device="cpu"):
    """Convert a TF batch (numpy) to PyTorch tensors, including IQL fields."""
    d = {
        "obs_image": torch.from_numpy(batch["observations"]["image"]),
        "goal_image": torch.from_numpy(batch["goals"]["image"]),
        "obs_proprio": torch.from_numpy(batch["observations"]["proprio"]),
        "next_obs_image": torch.from_numpy(batch["next_observations"]["image"]),
        "next_obs_proprio": torch.from_numpy(batch["next_observations"]["proprio"]),
        "actions": torch.from_numpy(batch["actions"]),
        "rewards": torch.from_numpy(batch["rewards"].astype("float32")),
        "masks": torch.from_numpy(batch["masks"].astype("float32")),
    }
    if "state256" in batch["observations"]:
        d["obs_state256"] = torch.from_numpy(batch["observations"]["state256"])
        d["next_obs_state256"] = torch.from_numpy(batch["next_observations"]["state256"])
    return d


def load_raw_trajectories(tfrecord_paths, n=3, seed=42, image_encoding="jpeg"):
    """Load n raw trajectories from TFRecords for visualization.

    Identical to gcbc_jax vis.load_vis_trajectories.
    """
    rng = np.random.RandomState(seed)
    n = min(n, len(tfrecord_paths))
    selected = rng.choice(len(tfrecord_paths), size=n, replace=False)

    img_dtype = tf.string if image_encoding == "jpeg" else tf.uint8

    trajectories = []
    for idx in sorted(selected):
        path = tfrecord_paths[idx]
        raw_dataset = tf.data.TFRecordDataset(path)
        for raw_record in raw_dataset:
            features = {k: tf.io.FixedLenFeature([], tf.string) for k in [
                "observations/images0", "observations/state",
                "actions", "goal_image",
            ]}
            parsed = tf.io.parse_single_example(raw_record, features)
            obs_raw = tf.io.parse_tensor(
                parsed["observations/images0"], img_dtype)
            goal_raw = tf.io.parse_tensor(
                parsed["goal_image"], img_dtype)
            if image_encoding == "jpeg":
                obs_images = _decode_jpeg_frames(obs_raw).numpy()
                goal_image = tf.io.decode_jpeg(goal_raw, channels=3).numpy()
            else:
                obs_images = obs_raw.numpy()
                goal_image = goal_raw.numpy()
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
