"""PyTorch eval policy wrapper for eval_ispatialgym.py.

Loads a trained PyTorch GCBCPolicy or GCDDPMBCPolicy checkpoint and provides
the forward(obs) -> torch.Tensor(23,) interface expected by OmniGibson.
"""

import glob
import json
import os

import numpy as np
import torch
from PIL import Image

from .diffusion_model import GCDDPMBCPolicy
from .model import GCBCPolicy
from .proprio import extract_proprio_np, normalize_proprio_bounds_np


class TorchGCBCEvalPolicy:
    """Wraps a PyTorch GCBCPolicy or GCDDPMBCPolicy for eval_ispatialgym.py."""

    def __init__(self, checkpoint_dir: str, goal_image_path: str,
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
            proprio_dim = 23

        # Load goal image
        goal_img = np.array(Image.open(goal_image_path).convert("RGB"))
        self.goal_image = goal_img  # uint8

        # Find and load latest checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
        assert ckpt_files, f"No checkpoints found in {checkpoint_dir}"
        ckpt_path = max(ckpt_files,
                        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Auto-detect policy type from checkpoint
        ckpt_args = ckpt.get("args", {})
        policy_type = ckpt_args.get("policy", "gcbc")

        if policy_type == "gc_ddpm_bc":
            self.model = GCDDPMBCPolicy(
                action_dim=action_dim,
                use_proprio=use_proprio,
                proprio_dim=proprio_dim,
                diffusion_steps=ckpt_args.get("diffusion_steps", 20),
            ).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.target_state_dict = ckpt.get("target_state_dict")
            print(f"Loaded PyTorch GCDDPMBCPolicy from {ckpt_path}")
        else:
            self.model = GCBCPolicy(
                action_dim=action_dim,
                use_proprio=use_proprio,
                proprio_dim=proprio_dim,
            ).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.target_state_dict = None
            print(f"Loaded PyTorch GCBCPolicy from {ckpt_path}")

        self.model.eval()
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
        head_rgb = obs[head_key]

        # Convert torch -> numpy, ensure RGB
        obs_image = head_rgb.cpu().numpy().astype(np.uint8)
        if obs_image.shape[-1] == 4:
            obs_image = obs_image[..., :3]

        # Resize obs to match goal resolution if needed
        goal_H, goal_W = self.goal_image.shape[:2]
        obs_H, obs_W = obs_image.shape[:2]
        if obs_H != goal_H or obs_W != goal_W:
            obs_image = np.array(
                Image.fromarray(obs_image).resize((goal_W, goal_H))
            )

        # Build tensors
        obs_t = torch.from_numpy(obs_image[np.newaxis]).to(self.device)
        goal_t = torch.from_numpy(self.goal_image[np.newaxis]).to(self.device)

        # Extract proprio
        if self.use_proprio:
            proprio_256 = obs["robot_r1::proprio"].cpu().numpy().astype(np.float32)
            proprio = extract_proprio_np(proprio_256, add_eef=self.add_eef_proprio)
            if self.normalize_proprio:
                proprio = normalize_proprio_bounds_np(proprio, add_eef=self.add_eef_proprio)
            proprio_t = torch.from_numpy(proprio[np.newaxis]).to(self.device)
        else:
            proprio_t = None

        # Run inference
        with torch.no_grad():
            actions = self.model.get_action(
                obs_t, goal_t, proprio_t, argmax=True,
                target_state_dict=self.target_state_dict,
            )

        # Denormalize
        actions_np = actions[0].cpu().numpy()
        actions_denorm = actions_np * self.action_std + self.action_mean

        return torch.tensor(actions_denorm, dtype=torch.float32)

    def reset(self):
        pass


def load_torch_gcbc_policy(checkpoint_dir: str, episode_dir: str,
                           use_proprio: bool = False, add_eef_proprio: bool = False,
                           normalize_proprio: bool = False) -> TorchGCBCEvalPolicy:
    """Convenience loader for eval_ispatialgym.py integration."""
    goal_path = os.path.join(episode_dir, "reference_image2.png")
    if not os.path.exists(goal_path):
        goal_path = os.path.join(episode_dir, "reference_image.png")
    return TorchGCBCEvalPolicy(
        checkpoint_dir, goal_path,
        use_proprio=use_proprio,
        add_eef_proprio=add_eef_proprio,
        normalize_proprio=normalize_proprio,
    )
