"""PyTorch Goal-Conditioned DDPM BC policy.

Port of the JAX gc_ddpm_bc agent from jaxrl_m:
  - ScoreActor with FourierFeatures time embedding
  - MLPResNet reverse (noise prediction) network
  - Cosine beta schedule, 20 diffusion steps
  - EMA target network for inference
  - Reuses ResNetV1Encoder from model.py (early goal concat)
"""

import math

import numpy as np
import torch
import torch.nn as nn

from .model import ResNetV1Encoder


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ"""
    steps = timesteps + 1
    t = np.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


# ---------------------------------------------------------------------------
# FourierFeatures (time embedding)
# ---------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    """Learnable Fourier feature embedding for diffusion timesteps.

    Matches jaxrl_m FourierFeatures with learnable=True.
    """

    def __init__(self, output_size, learnable=True):
        super().__init__()
        self.output_size = output_size
        self.learnable = learnable
        if learnable:
            # Shape (output_size//2, 1) matching JAX: normal(0.2)
            self.W = nn.Parameter(torch.randn(output_size // 2, 1) * 0.2)
        else:
            half_dim = output_size // 2
            f = math.log(10000) / (half_dim - 1)
            freqs = torch.exp(torch.arange(half_dim).float() * -f)
            self.register_buffer("freqs", freqs)

    def forward(self, x):
        # x: (B, 1)
        if self.learnable:
            f = 2 * math.pi * x @ self.W.T  # (B, output_size//2)
        else:
            f = x * self.freqs  # (B, output_size//2)
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)  # (B, output_size)


# ---------------------------------------------------------------------------
# MLPResNet (reverse/noise-prediction network)
# ---------------------------------------------------------------------------

class MLPResNetBlock(nn.Module):
    """Residual MLP block matching jaxrl_m MLPResNetBlock.

    Order: Dropout -> LayerNorm -> Dense(4x) -> SiLU -> Dense(1x) + residual
    """

    def __init__(self, features, dropout_rate=None, use_layer_norm=False):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate

        if dropout_rate is not None and dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(features)

        self.dense1 = nn.Linear(features, features * 4)
        self.dense2 = nn.Linear(features * 4, features)
        # Residual projection if input dim != features (used by first block)
        self.residual_proj = None  # set dynamically on first forward

    def forward(self, x, train=True):
        residual = x

        if self.dropout is not None:
            self.dropout.train(train)
            x = self.dropout(x)

        if self.use_layer_norm:
            x = self.layer_norm(x)

        x = self.dense1(x)
        x = torch.nn.functional.silu(x)
        x = self.dense2(x)

        # Residual projection if shapes differ
        if residual.shape[-1] != x.shape[-1]:
            if self.residual_proj is None:
                self.residual_proj = nn.Linear(
                    residual.shape[-1], x.shape[-1], device=x.device
                )
            residual = self.residual_proj(residual)

        return residual + x


class MLPResNet(nn.Module):
    """MLP with residual blocks matching jaxrl_m MLPResNet.

    Dense(hidden_dim) -> N x MLPResNetBlock -> SiLU -> Dense(out_dim)
    """

    def __init__(self, num_blocks, out_dim, in_dim, dropout_rate=None,
                 use_layer_norm=False, hidden_dim=256):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            MLPResNetBlock(hidden_dim, dropout_rate, use_layer_norm)
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, train=True):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x, train=train)
        x = torch.nn.functional.silu(x)
        x = self.output_proj(x)
        return x


# ---------------------------------------------------------------------------
# GCDDPMBCPolicy
# ---------------------------------------------------------------------------

class GCDDPMBCPolicy(nn.Module):
    """Goal-Conditioned DDPM Behavioral Cloning policy.

    Matches the JAX GCDDPMBCAgent architecture:
        1. Normalize images: uint8 / 127.5 - 1.0
        2. Early goal concat: cat(obs, goal) on channel dim -> 6ch
        3. ResNet encoder -> 512-dim
        4. Optional proprio concat -> 512 + proprio_dim
        5. ScoreActor: FourierFeatures(time) + CondEncoder + concat(cond, obs_enc, actions)
        6. MLPResNet reverse network -> noise prediction
        7. DDPM training loss: MSE(predicted_noise, true_noise)
        8. Reverse diffusion sampling with EMA target network
    """

    def __init__(self, action_dim=23, use_proprio=False, proprio_dim=23,
                 diffusion_steps=20, beta_schedule="cosine",
                 time_dim=32, num_blocks=3, hidden_dim=256,
                 dropout_rate=0.1, use_layer_norm=True):
        super().__init__()
        self.action_dim = action_dim
        self.use_proprio = use_proprio
        self.diffusion_steps = diffusion_steps

        # Vision encoder (shared with GCBCPolicy)
        self.encoder = ResNetV1Encoder(
            in_channels=6,
            add_spatial_coordinates=True,
        )
        encoder_out_dim = self.encoder.output_dim  # 512
        if use_proprio:
            encoder_out_dim += proprio_dim

        # Time embedding: FourierFeatures -> CondEncoder MLP
        self.fourier_features = FourierFeatures(time_dim, learnable=True)
        # MLP(hidden_dims=(2*time_dim, time_dim)) with activate_final=False
        # = Dense(2*time_dim) -> SiLU -> Dense(time_dim)
        self.cond_encoder = nn.Sequential(
            nn.Linear(time_dim, 2 * time_dim),
            nn.SiLU(),
            nn.Linear(2 * time_dim, time_dim),
        )

        # Reverse network (noise prediction)
        # Input dim = time_dim + encoder_out_dim + action_dim
        reverse_in_dim = time_dim + encoder_out_dim + action_dim
        self.reverse_network = MLPResNet(
            num_blocks=num_blocks,
            out_dim=action_dim,
            in_dim=reverse_in_dim,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
            hidden_dim=hidden_dim,
        )

        # Precompute noise schedule
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(diffusion_steps)
        elif beta_schedule == "linear":
            betas = np.linspace(1e-4, 2e-2, diffusion_steps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alpha_hats = np.array([np.prod(alphas[:i + 1]) for i in range(diffusion_steps)])

        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alphas", torch.tensor(alphas, dtype=torch.float32))
        self.register_buffer("alpha_hats", torch.tensor(alpha_hats, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        """Initialize MLP weights with xavier_uniform (matching JAX default_init)."""
        for m in self.cond_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.reverse_network.input_proj.weight)
        nn.init.zeros_(self.reverse_network.input_proj.bias)
        nn.init.xavier_uniform_(self.reverse_network.output_proj.weight)
        nn.init.zeros_(self.reverse_network.output_proj.bias)

    def _encode_obs_goal(self, obs_image, goal_image, proprio=None):
        """Encode observation + goal images (and optional proprio).

        Args:
            obs_image: (B, H, W, 3) uint8
            goal_image: (B, H, W, 3) uint8
            proprio: (B, proprio_dim) float, optional

        Returns:
            encoding: (B, encoder_out_dim)
        """
        obs = obs_image.float() / 127.5 - 1.0
        goal = goal_image.float() / 127.5 - 1.0
        obs = obs.permute(0, 3, 1, 2)   # NHWC -> NCHW
        goal = goal.permute(0, 3, 1, 2)
        x = torch.cat([obs, goal], dim=1)  # (B, 6, H, W)
        encoding = self.encoder(x)  # (B, 512)

        if self.use_proprio and proprio is not None:
            encoding = torch.cat([encoding, proprio], dim=-1)

        return encoding

    def _predict_noise(self, obs_enc, noisy_actions, time, train=True):
        """Predict noise given encoded obs, noisy actions, and timestep.

        Matches ScoreActor.__call__ from diffusion_nets.py.

        Args:
            obs_enc: (B, encoder_out_dim) from _encode_obs_goal
            noisy_actions: (B, action_dim)
            time: (B, 1) float timestep
            train: enable dropout

        Returns:
            eps_pred: (B, action_dim) predicted noise
        """
        t_ff = self.fourier_features(time)       # (B, time_dim)
        cond_enc = self.cond_encoder(t_ff)        # (B, time_dim)
        reverse_input = torch.cat([cond_enc, obs_enc, noisy_actions], dim=-1)
        return self.reverse_network(reverse_input, train=train)

    def compute_loss(self, obs_image, goal_image, actions, proprio=None):
        """Compute DDPM BC loss.

        Matches GCDDPMBCAgent.update from gc_ddpm_bc.py.

        Args:
            obs_image: (B, H, W, 3) uint8
            goal_image: (B, H, W, 3) uint8
            actions: (B, action_dim) float, z-score normalized
            proprio: (B, proprio_dim) float, optional

        Returns:
            loss: scalar
            metrics: dict
        """
        B = actions.shape[0]
        obs_enc = self._encode_obs_goal(obs_image, goal_image, proprio)

        # Sample random timesteps
        time = torch.randint(0, self.diffusion_steps, (B,), device=actions.device)

        # Sample noise
        noise = torch.randn_like(actions)

        # Forward diffusion: x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1 - alpha_hat_t) * eps
        ah = self.alpha_hats[time]  # (B,)
        sqrt_ah = torch.sqrt(ah)[:, None]           # (B, 1)
        sqrt_one_minus_ah = torch.sqrt(1 - ah)[:, None]  # (B, 1)
        noisy_actions = sqrt_ah * actions + sqrt_one_minus_ah * noise

        # Predict noise
        eps_pred = self._predict_noise(
            obs_enc, noisy_actions, time[:, None].float(), train=True
        )

        # DDPM loss: MSE over action dims, mean over batch
        ddpm_loss = (eps_pred - noise).pow(2).sum(dim=-1)  # (B,)
        loss = ddpm_loss.mean()

        metrics = {
            "ddpm_loss": loss.item(),
            "mse": ddpm_loss.mean().item(),
        }
        return loss, metrics

    @torch.no_grad()
    def sample_actions(self, obs_enc, predict_noise_fn, temperature=1.0,
                       clip_sampler=True):
        """Reverse diffusion sampling.

        Matches GCDDPMBCAgent.sample_actions from gc_ddpm_bc.py.

        Args:
            obs_enc: (B, encoder_out_dim)
            predict_noise_fn: callable(obs_enc, noisy_actions, time, train=False)
            temperature: noise scaling (default 1.0)
            clip_sampler: clip actions to [-2, 2]

        Returns:
            actions: (B, action_dim)
        """
        B = obs_enc.shape[0]
        device = obs_enc.device

        # Start from pure noise
        current_x = torch.randn(B, self.action_dim, device=device)

        # Reverse loop: t = T-1, T-2, ..., 1, 0
        for t in range(self.diffusion_steps - 1, -1, -1):
            t_tensor = torch.full((B, 1), t, device=device, dtype=torch.float32)

            eps_pred = predict_noise_fn(obs_enc, current_x, t_tensor, train=False)

            alpha_1 = 1.0 / torch.sqrt(self.alphas[t])
            alpha_2 = (1.0 - self.alphas[t]) / torch.sqrt(1.0 - self.alpha_hats[t])
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            if t > 0:
                z = torch.randn_like(current_x)
                current_x = current_x + torch.sqrt(self.betas[t]) * temperature * z

            if clip_sampler:
                current_x = current_x.clamp(-2.0, 2.0)

        return current_x

    def get_action(self, obs_image, goal_image, proprio=None, argmax=True,
                   target_state_dict=None):
        """Get action for inference (no grad, eval mode).

        Unified interface matching GCBCPolicy.get_action.

        Args:
            obs_image: (B, H, W, 3) uint8
            goal_image: (B, H, W, 3) uint8
            proprio: optional (B, D)
            argmax: accepted for API compat (ignored, diffusion always samples)
            target_state_dict: EMA target parameters for inference

        Returns:
            actions: (B, action_dim)
        """
        with torch.no_grad():
            obs_enc = self._encode_obs_goal(obs_image, goal_image, proprio)

            if target_state_dict is not None:
                # Swap in target params for noise prediction
                original_state = {k: v.clone() for k, v in self.state_dict().items()}
                self.load_state_dict(target_state_dict)
                actions = self.sample_actions(obs_enc, self._predict_noise)
                self.load_state_dict(original_state)
            else:
                actions = self.sample_actions(obs_enc, self._predict_noise)

        return actions
