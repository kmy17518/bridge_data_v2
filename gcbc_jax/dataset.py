"""Custom BridgeDataset subclass that uses fixed goal images per episode.

Instead of relabeling goals from future trajectory states (uniform strategy),
this dataset stores a fixed reference image per trajectory and uses it as
the goal for all transitions in that trajectory.
"""

import tensorflow as tf

from jaxrl_m.data.bridge_dataset import BridgeDataset

from .proprio import extract_proprio_tf, normalize_proprio_bounds_tf


class FixedGoalBridgeDataset(BridgeDataset):
    """BridgeDataset with fixed goal images stored in TFRecords.

    Each TFRecord contains a 'goal_image' field (uint8 image) that is used
    as the goal for all transitions, instead of sampling from future states.

    Extra kwargs (passed through to __init__):
        use_proprio: If True, extract 23 or 37 dim proprio from 256-dim state.
            If False, proprio is passed through as-is (and ignored by the agent).
        add_eef_proprio: If True (and use_proprio=True), append 14 EEF dims
            (left/right pos + quat) → 37-dim total.
        normalize_proprio: If True (and use_proprio=True), normalize proprio
            to [-1, 1] using JOINT_RANGE / EEF_POSITION_RANGE bounds.
    """

    # Extend PROTO_TYPE_SPEC with the goal_image field
    PROTO_TYPE_SPEC = {
        **BridgeDataset.PROTO_TYPE_SPEC,
        "goal_image": tf.uint8,
    }

    def __init__(self, *args, use_proprio=False, add_eef_proprio=False,
                 normalize_proprio=False, **kwargs):
        self.use_proprio = use_proprio
        self.add_eef_proprio = add_eef_proprio
        self.normalize_proprio = normalize_proprio
        super().__init__(*args, **kwargs)

    def _decode_example(self, example_proto):
        """Decode example including the fixed goal image."""
        features = {
            key: tf.io.FixedLenFeature([], tf.string)
            for key in self.PROTO_TYPE_SPEC.keys()
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        parsed_tensors = {
            key: tf.io.parse_tensor(parsed_features[key], dtype)
            for key, dtype in self.PROTO_TYPE_SPEC.items()
        }

        obs_proprio = parsed_tensors["observations/state"]
        next_obs_proprio = parsed_tensors["next_observations/state"]

        # Extract 23 or 37 dim proprio from 256-dim state
        if self.use_proprio:
            obs_proprio = extract_proprio_tf(
                tf.cast(obs_proprio, tf.float32), add_eef=self.add_eef_proprio)
            next_obs_proprio = extract_proprio_tf(
                tf.cast(next_obs_proprio, tf.float32), add_eef=self.add_eef_proprio)

        return {
            "observations": {
                "image": parsed_tensors["observations/images0"],
                "proprio": obs_proprio,
            },
            "next_observations": {
                "image": parsed_tensors["next_observations/images0"],
                "proprio": next_obs_proprio,
            },
            "actions": parsed_tensors["actions"],
            "terminals": parsed_tensors["terminals"],
            "truncates": parsed_tensors["truncates"],
            "goal_image": parsed_tensors["goal_image"],  # (H, W, 3) uint8
        }

    def _add_goals(self, traj):
        """Use the fixed goal image instead of future-state relabeling."""
        traj_len = tf.shape(traj["actions"])[0]

        # Broadcast the single goal image to all timesteps
        goal_image = tf.broadcast_to(
            traj["goal_image"][tf.newaxis],  # (1, H, W, 3)
            [traj_len, tf.shape(traj["goal_image"])[0],
             tf.shape(traj["goal_image"])[1], 3]
        )

        traj["goals"] = {
            "image": goal_image,
            "proprio": traj["observations"]["proprio"],  # dummy, not used
        }

        # Reward: -1 for all (never "reached" the goal via relabeling)
        traj["rewards"] = tf.cast(tf.fill([traj_len], -1), tf.int32)
        traj["masks"] = tf.logical_not(traj["terminals"])

        # Clean up the raw goal_image field
        del traj["goal_image"]

        return traj

    def _process_actions(self, traj):
        """Skip action relabeling (our actions are already correct).

        Normalizes actions via z-score (action_proprio_metadata).
        Normalizes proprio via bounds (JOINT_RANGE) if normalize_proprio=True,
        otherwise leaves proprio raw.
        """
        if self.action_proprio_metadata is not None:
            if self.normalization_type == "normal":
                # Always z-score normalize actions
                traj["actions"] = (
                    traj["actions"] - self.action_proprio_metadata["action"]["mean"]
                ) / self.action_proprio_metadata["action"]["std"]

        # Proprio normalization (bounds, not z-score)
        if self.use_proprio and self.normalize_proprio:
            for key in ["observations", "next_observations"]:
                traj[key]["proprio"] = normalize_proprio_bounds_tf(
                    traj[key]["proprio"], add_eef=self.add_eef_proprio)

        return traj
