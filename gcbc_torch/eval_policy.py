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
from .iql_model import GCIQLPolicy
from .model import GCBCPolicy
from .proprio import (
    extract_proprio_np, normalize_proprio_bounds_np,
    denormalize_actions_bounds_np, ACTION_BOUNDS_LOW_23,
    apply_ablation_to_proprio, ablation_uses_image, ablation_uses_proprio,
    ablation_proprio_dim, parse_base_proprio_keys,
)


class TorchGCBCEvalPolicy:
    """Wraps a PyTorch GCBCPolicy or GCDDPMBCPolicy for eval_ispatialgym.py."""

    def __init__(self, checkpoint_dir: str, goal_image_path: str,
                 use_proprio: bool = False, add_eef_proprio: bool = False,
                 normalize_proprio: bool = False,
                 action_metadata_path: str | None = None,
                 image_size: int = 256,
                 proprio_ablation_mode: str | None = None,
                 base_proprio_keys: str | None = None):
        self.use_proprio = use_proprio
        self.add_eef_proprio = add_eef_proprio
        self.normalize_proprio = normalize_proprio
        self.action_metadata_path = action_metadata_path
        self.image_size = image_size
        self.proprio_ablation_mode = proprio_ablation_mode

        # Ablation mode overrides
        self._use_ablation = proprio_ablation_mode is not None
        if self._use_ablation:
            self._base_keys = parse_base_proprio_keys(
                base_proprio_keys if base_proprio_keys is not None else "base_qvel"
            )
            self._ablation_use_image = ablation_uses_image(proprio_ablation_mode)
            self._ablation_use_proprio = ablation_uses_proprio(proprio_ablation_mode)
            # Ring buffer (capacity 64) for shuffled modes
            _buf_dim = 37  # always store full 37-dim normalized
            self._eval_proprio_buffer = np.zeros((64, _buf_dim), dtype=np.float32)
            self._eval_buf_count = 0  # total entries pushed so far

        # Load z-score stats for legacy checkpoints
        if action_metadata_path is not None:
            with open(action_metadata_path) as f:
                metadata = json.load(f)
            self.action_mean = np.array(metadata["action"]["mean"], dtype=np.float32)
            self.action_std = np.array(metadata["action"]["std"], dtype=np.float32)
            print(f"Loaded z-score action metadata from {action_metadata_path}")

        action_dim = len(ACTION_BOUNDS_LOW_23)  # 23 for R1Pro

        # Determine proprio dimension
        if self._use_ablation:
            # _base_keys set above; if loading from checkpoint, may override below
            proprio_dim = ablation_proprio_dim(proprio_ablation_mode, self._base_keys)
            use_proprio_model = self._ablation_use_proprio
        elif use_proprio:
            proprio_dim = 37 if add_eef_proprio else 23
            use_proprio_model = True
        else:
            proprio_dim = 23
            use_proprio_model = False

        # Load goal image — resize to match training resolution (convert_to_tfrecord.py)
        goal_img = np.array(
            Image.open(goal_image_path).convert("RGB").resize((image_size, image_size))
        )
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
        self._proprio_noise_std = float(ckpt_args.get("proprio_noise_std", 1.0))
        if self._use_ablation and ckpt_args.get("base_proprio_keys"):
            self._base_keys = parse_base_proprio_keys(ckpt_args["base_proprio_keys"])
            proprio_dim = ablation_proprio_dim(proprio_ablation_mode, self._base_keys)
        policy_type = ckpt_args.get("policy", "gcbc")

        if policy_type == "gc_iql":
            self.model = GCIQLPolicy(
                action_dim=action_dim,
                use_proprio=use_proprio_model,
                proprio_dim=proprio_dim,
            ).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.target_state_dict = ckpt.get("target_state_dict")
            print(f"Loaded PyTorch GCIQLPolicy from {ckpt_path}")
        elif policy_type == "gc_ddpm_bc":
            self.model = GCDDPMBCPolicy(
                action_dim=action_dim,
                use_proprio=use_proprio_model,
                proprio_dim=proprio_dim,
                diffusion_steps=ckpt_args.get("diffusion_steps", 20),
            ).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.target_state_dict = ckpt.get("target_state_dict")
            print(f"Loaded PyTorch GCDDPMBCPolicy from {ckpt_path}")
        else:
            # Detect encoder from checkpoint (default resnet for old checkpoints)
            encoder = ckpt_args.get("encoder", "resnetv1-34-bridge")
            encoder_config_dict = ckpt.get("encoder_config")
            _ablation_use_image_flag = (
                self._ablation_use_image if self._use_ablation else True
            )
            self.model = GCBCPolicy(
                action_dim=action_dim,
                use_proprio=use_proprio_model,
                proprio_dim=proprio_dim,
                encoder=encoder,
                encoder_model_name_or_path=ckpt_args.get("encoder_model_name_or_path"),
                train_encoder=ckpt_args.get("train_encoder", False),
                load_pretrained_weights=False,
                encoder_config_dict=encoder_config_dict,
                use_image=_ablation_use_image_flag,
            ).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.target_state_dict = None
            print(f"Loaded PyTorch GCBCPolicy from {ckpt_path} (encoder={encoder})")

        self.model.eval()
        if self._use_ablation:
            print(f"  Ablation mode: {proprio_ablation_mode} | "
                  f"use_image={self._ablation_use_image} use_proprio={self._ablation_use_proprio} "
                  f"proprio_dim={proprio_dim}")
        elif use_proprio:
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

        # Resize obs to match training resolution (convert_to_tfrecord.py)
        sz = self.image_size
        if obs_image.shape[0] != sz or obs_image.shape[1] != sz:
            obs_image = np.array(
                Image.fromarray(obs_image).resize((sz, sz))
            )

        # Build tensors
        obs_t = torch.from_numpy(obs_image[np.newaxis]).to(self.device)
        goal_t = torch.from_numpy(self.goal_image[np.newaxis]).to(self.device)

        # Extract proprio
        if self._use_ablation:
            proprio_256 = obs["robot_r1::proprio"].cpu().numpy().astype(np.float32)
            # Always extract and normalize full 37-dim for ablation
            proprio_37 = extract_proprio_np(proprio_256, add_eef=True)
            proprio_37 = normalize_proprio_bounds_np(proprio_37, add_eef=True)
            # Push into ring buffer
            buf_idx = self._eval_buf_count % 64
            self._eval_proprio_buffer[buf_idx] = proprio_37
            self._eval_buf_count += 1

            if self.proprio_ablation_mode.startswith("shuffled_") and self._eval_buf_count > 1:
                # Pop a random past entry from the buffer
                filled = min(self._eval_buf_count - 1, 64)  # exclude current
                rng = np.random.default_rng()
                rand_idx = rng.integers(0, filled)
                # Map to actual buffer slot (most recent is buf_idx, go back rand_idx+1)
                pick_idx = (buf_idx - 1 - rand_idx) % 64
                proprio_37_to_use = self._eval_proprio_buffer[pick_idx]
            else:
                proprio_37_to_use = proprio_37

            proprio_t_full = torch.from_numpy(proprio_37_to_use[np.newaxis]).to(self.device)
            if self._ablation_use_proprio:
                raw_state_t = torch.from_numpy(
                    proprio_256[np.newaxis]).to(self.device, dtype=torch.float32)
                proprio_t = apply_ablation_to_proprio(
                    proprio_t_full,
                    self.proprio_ablation_mode,
                    raw_state_256=raw_state_t,
                    base_keys=self._base_keys,
                    noise_std=self._proprio_noise_std,
                )
            else:
                proprio_t = None
        elif self.use_proprio:
            proprio_256 = obs["robot_r1::proprio"].cpu().numpy().astype(np.float32)
            proprio = extract_proprio_np(proprio_256, add_eef=self.add_eef_proprio)
            if self.normalize_proprio:
                proprio = normalize_proprio_bounds_np(proprio, add_eef=self.add_eef_proprio)
            if self.action_metadata_path is not None:
                # Legacy order: base(3)+trunk(4)+arm_l(7)+arm_r(7)+grip_l(1)+grip_r(1)
                # Current order: base(3)+trunk(4)+arm_l(7)+grip_l(1)+arm_r(7)+grip_r(1)
                proprio = np.concatenate([
                    proprio[..., :14],   # base(3)+trunk(4)+arm_left(7)
                    proprio[..., 15:22], # arm_right(7)
                    proprio[..., 14:15], # gripper_left(1)
                    proprio[..., 22:23], # gripper_right(1)
                ], axis=-1)
            proprio_t = torch.from_numpy(proprio[np.newaxis]).to(self.device)
        else:
            proprio_t = None

        # Run inference
        with torch.no_grad():
            actions = self.model.get_action(
                obs_t, goal_t, proprio_t, argmax=True,
                target_state_dict=self.target_state_dict,
            )

        # Denormalize: z-score (legacy) or fixed joint-range bounds (default)
        actions_np = actions[0].cpu().numpy()
        if self.action_metadata_path is not None:
            actions_denorm = actions_np * self.action_std + self.action_mean
        else:
            actions_denorm = denormalize_actions_bounds_np(actions_np)

        return torch.tensor(actions_denorm, dtype=torch.float32)

    def preprocess_obs_for_comparison(self, obs):
        """Return preprocessed (obs_image, goal_image, proprio) without inference.

        Replicates the same preprocessing as forward() but returns numpy arrays
        instead of running the model.

        Returns:
            dict with:
                "obs_image": (image_size, image_size, 3) uint8 numpy
                "goal_image": (image_size, image_size, 3) uint8 numpy
                "proprio": (23,) or (37,) float32 numpy, or None
        """
        head_key = "robot_r1::robot_r1:zed_link:Camera:0::rgb"
        head_rgb = obs[head_key]

        obs_image = head_rgb.cpu().numpy().astype(np.uint8)
        if obs_image.shape[-1] == 4:
            obs_image = obs_image[..., :3]

        sz = self.image_size
        if obs_image.shape[0] != sz or obs_image.shape[1] != sz:
            obs_image = np.array(
                Image.fromarray(obs_image).resize((sz, sz))
            )

        proprio = None
        if self.use_proprio:
            proprio_256 = obs["robot_r1::proprio"].cpu().numpy().astype(np.float32)
            proprio = extract_proprio_np(proprio_256, add_eef=self.add_eef_proprio)
            if self.normalize_proprio:
                proprio = normalize_proprio_bounds_np(proprio, add_eef=self.add_eef_proprio)
            if self.action_metadata_path is not None:
                proprio = np.concatenate([
                    proprio[..., :14],
                    proprio[..., 15:22],
                    proprio[..., 14:15],
                    proprio[..., 22:23],
                ], axis=-1)

        return {
            "obs_image": obs_image,
            "goal_image": self.goal_image.copy(),
            "proprio": proprio,
        }

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
