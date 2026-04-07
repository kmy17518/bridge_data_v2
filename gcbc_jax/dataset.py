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

    # Extend PROTO_TYPE_SPEC with the goal_image field.
    # Image fields store JPEG-compressed string tensors.
    PROTO_TYPE_SPEC = {
        **{k: (tf.string if v == tf.uint8 else v)
           for k, v in BridgeDataset.PROTO_TYPE_SPEC.items()},
        "goal_image": tf.string,
    }

    def __init__(self, *args, use_proprio=False, add_eef_proprio=False,
                 normalize_proprio=False, **kwargs):
        self.use_proprio = use_proprio
        self.add_eef_proprio = add_eef_proprio
        self.normalize_proprio = normalize_proprio
        super().__init__(*args, **kwargs)

    @staticmethod
    def _decode_jpeg_frames(jpeg_tensor):
        """Decode a 1-D string tensor of JPEG bytes into (T, H, W, 3) uint8."""
        return tf.map_fn(
            lambda j: tf.io.decode_jpeg(j, channels=3),
            jpeg_tensor,
            fn_output_signature=tf.uint8,
        )

    def _decode_example(self, example_proto):
        """Decode example including the fixed goal image (JPEG-compressed)."""
        features = {
            key: tf.io.FixedLenFeature([], tf.string)
            for key in self.PROTO_TYPE_SPEC.keys()
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        parsed_tensors = {
            key: tf.io.parse_tensor(parsed_features[key], dtype)
            for key, dtype in self.PROTO_TYPE_SPEC.items()
        }

        obs_images = self._decode_jpeg_frames(parsed_tensors["observations/images0"])
        next_obs_images = self._decode_jpeg_frames(
            parsed_tensors["next_observations/images0"])
        goal_image = tf.io.decode_jpeg(parsed_tensors["goal_image"], channels=3)

        obs_proprio = parsed_tensors["observations/state"]
        next_obs_proprio = parsed_tensors["next_observations/state"]

        if self.use_proprio:
            obs_proprio = extract_proprio_tf(
                tf.cast(obs_proprio, tf.float32), add_eef=self.add_eef_proprio)
            next_obs_proprio = extract_proprio_tf(
                tf.cast(next_obs_proprio, tf.float32), add_eef=self.add_eef_proprio)

        return {
            "observations": {
                "image": obs_images,
                "proprio": obs_proprio,
            },
            "next_observations": {
                "image": next_obs_images,
                "proprio": next_obs_proprio,
            },
            "actions": parsed_tensors["actions"],
            "terminals": parsed_tensors["terminals"],
            "truncates": parsed_tensors["truncates"],
            "goal_image": goal_image,
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
