"""JAX→PyTorch eval policy wrapper for eval_ispatialgym.py.

Loads a trained JAX GCBCAgent checkpoint and provides the
forward(obs) -> torch.Tensor(23,) interface expected by OmniGibson.
"""

import glob
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import serialization
from PIL import Image

from jaxrl_m.agents.continuous.gc_bc import GCBCAgent
from jaxrl_m.vision import encoders

from .proprio import extract_proprio_np, normalize_proprio_bounds_np


class JAXGCBCEvalPolicy:
    """Wraps a JAX GCBCAgent for use with eval_ispatialgym.py."""

    def __init__(self, checkpoint_dir: str, goal_image_path: str,
                 encoder_name: str = "resnetv1-34-bridge",
                 use_proprio: bool = False, add_eef_proprio: bool = False,
                 normalize_proprio: bool = False):
        self.use_proprio = use_proprio
        self.add_eef_proprio = add_eef_proprio
        self.normalize_proprio = normalize_proprio

        # Load action normalization stats
        stats_path = os.path.join(checkpoint_dir, "action_proprio_metadata.json")
        with open(stats_path) as f:
            self.metadata = json.load(f)
        self.action_mean = np.array(self.metadata["action"]["mean"], dtype=np.float32)
        self.action_std = np.array(self.metadata["action"]["std"], dtype=np.float32)
        action_dim = len(self.action_mean)

        # Determine proprio dimension
        if use_proprio:
            proprio_dim = 37 if add_eef_proprio else 23
        else:
            proprio_dim = action_dim  # dummy, not used by agent

        # Create encoder and dummy agent for checkpoint restoration
        encoder_def = encoders[encoder_name](
            pooling_method="avg",
            add_spatial_coordinates=True,
            act="swish",
        )

        # Load goal image to determine image shape
        goal_img = np.array(Image.open(goal_image_path).convert("RGB"))  # (H, W, 3)
        self.goal_image = goal_img  # uint8

        # Create dummy inputs matching checkpoint shapes
        H, W = goal_img.shape[:2]
        dummy_obs = {
            "image": np.zeros((1, H, W, 3), dtype=np.uint8),
            "proprio": np.zeros((1, proprio_dim), dtype=np.float32),
        }
        dummy_goals = {
            "image": np.zeros((1, H, W, 3), dtype=np.uint8),
            "proprio": np.zeros((1, proprio_dim), dtype=np.float32),
        }
        dummy_actions = np.zeros((1, action_dim), dtype=np.float32)

        rng = jax.random.PRNGKey(0)
        agent = GCBCAgent.create(
            rng=rng,
            observations=dummy_obs,
            goals=dummy_goals,
            actions=dummy_actions,
            encoder_def=encoder_def,
            early_goal_concat=True,
            shared_goal_encoder=True,
            use_proprio=use_proprio,
            network_kwargs=dict(hidden_dims=(256, 256, 256), dropout_rate=0.1),
            policy_kwargs=dict(
                tanh_squash_distribution=False,
                fixed_std=[1.0] * action_dim,
                state_dependent_std=False,
            ),
            learning_rate=3e-4,
            warmup_steps=2000,
            decay_steps=100000,
        )

        # Restore checkpoint (find latest checkpoint_* file)
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*"))
        ckpt_files = [f for f in ckpt_files if os.path.isfile(f)]
        assert ckpt_files, f"No checkpoints found in {checkpoint_dir}"
        ckpt_path = max(ckpt_files, key=lambda p: int(p.rsplit("_", 1)[-1]))
        with open(ckpt_path, "rb") as f:
            self.agent = serialization.from_bytes(agent, f.read())
        self.rng = jax.random.PRNGKey(42)
        print(f"Loaded JAX GCBCAgent from {ckpt_path}")
        if use_proprio:
            print(f"  Proprio: {proprio_dim}-dim (add_eef={add_eef_proprio}, "
                  f"normalize={normalize_proprio})")

    def forward(self, obs, *args, **kwargs):
        """Policy interface for eval_ispatialgym.py.

        Args:
            obs: dict with "robot_r1::robot_r1:zed_link:Camera:0::rgb",
                 "robot_r1::proprio" (256-dim), etc.

        Returns:
            torch.Tensor: (23,) action
        """
        # Extract head camera image
        head_key = "robot_r1::robot_r1:zed_link:Camera:0::rgb"
        head_rgb = obs[head_key]  # torch tensor (H, W, 3) uint8

        # Convert torch → numpy, ensure RGB (drop alpha if RGBA)
        obs_image = head_rgb.cpu().numpy().astype(np.uint8)
        if obs_image.shape[-1] == 4:
            obs_image = obs_image[..., :3]

        # Resize obs to match goal image resolution if needed
        goal_H, goal_W = self.goal_image.shape[:2]
        obs_H, obs_W = obs_image.shape[:2]
        if obs_H != goal_H or obs_W != goal_W:
            obs_image = np.array(
                Image.fromarray(obs_image).resize((goal_W, goal_H))
            )

        # Extract proprio
        if self.use_proprio:
            proprio_256 = obs["robot_r1::proprio"].cpu().numpy().astype(np.float32)
            proprio = extract_proprio_np(proprio_256, add_eef=self.add_eef_proprio)
            if self.normalize_proprio:
                proprio = normalize_proprio_bounds_np(proprio, add_eef=self.add_eef_proprio)
            proprio_jnp = jnp.array(proprio[np.newaxis])  # (1, 23 or 37)
        else:
            proprio_dim = len(self.action_mean)
            proprio_jnp = jnp.zeros((1, proprio_dim), dtype=jnp.float32)

        # Build JAX observation and goal dicts
        jax_obs = {
            "image": jnp.array(obs_image[np.newaxis]),  # (1, H, W, 3) uint8
            "proprio": proprio_jnp,
        }
        jax_goals = {
            "image": jnp.array(self.goal_image[np.newaxis]),  # (1, H, W, 3) uint8
            "proprio": jnp.zeros_like(proprio_jnp),
        }

        # Run JAX inference (deterministic)
        self.rng, sample_rng = jax.random.split(self.rng)
        actions = self.agent.sample_actions(
            observations=jax_obs,
            goals=jax_goals,
            seed=sample_rng,
            temperature=1.0,
            argmax=True,
        )

        # Denormalize
        actions_np = np.array(actions[0])  # (action_dim,)
        actions_denorm = actions_np * self.action_std + self.action_mean

        return torch.tensor(actions_denorm, dtype=torch.float32)

    def reset(self):
        pass


def load_jax_gcbc_policy(checkpoint_dir: str, episode_dir: str,
                         use_proprio: bool = False, add_eef_proprio: bool = False,
                         normalize_proprio: bool = False) -> JAXGCBCEvalPolicy:
    """Convenience loader for eval_ispatialgym.py integration."""
    # Try reference_image2.png first, fall back to reference_image.png
    goal_path = os.path.join(episode_dir, "reference_image2.png")
    if not os.path.exists(goal_path):
        goal_path = os.path.join(episode_dir, "reference_image.png")
    return JAXGCBCEvalPolicy(
        checkpoint_dir, goal_path,
        use_proprio=use_proprio,
        add_eef_proprio=add_eef_proprio,
        normalize_proprio=normalize_proprio,
    )
