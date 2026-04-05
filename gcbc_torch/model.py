"""PyTorch ResNetV1-34-Bridge encoder + GC-BC policy.

Exact reimplementation of the JAX/Flax architecture from jaxrl_m:
  - ResNetV1-34 with GroupNorm(4), Swish activation, spatial coordinates
  - Early goal concatenation (obs+goal channel-wise before encoder)
  - Average spatial pooling -> 512-dim
  - Optional proprioception concatenation
  - 3-layer MLP (256-256-256) with SiLU and 0.1 dropout
  - Fixed std=1.0, MultivariateNormal output
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AddSpatialCoordinates(nn.Module):
    """Append normalized (x, y) coordinate channels to input tensor."""

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # Normalized to [-1, 1] matching JAX: np.arange(s) / (s - 1) * 2 - 1
        ys = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # (1, 2, H, W) -> broadcast to (B, 2, H, W)
        coords = torch.stack([grid_y, grid_x], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return torch.cat([x, coords], dim=1)


class MyGroupNorm(nn.GroupNorm):
    """GroupNorm matching jaxrl_m's MyGroupNorm (handles 3D input)."""

    def __init__(self, num_channels, num_groups=4, eps=1e-5):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps)


class ResNetBlock(nn.Module):
    """ResNet basic block with GroupNorm.

    Matches jaxrl_m ResNetBlock:
        Conv3x3(stride) -> GN -> act -> Conv3x3 -> GN
        + shortcut: Conv1x1(stride) -> GN if shape changes
        -> act(residual + y)
    """

    def __init__(self, in_channels, out_channels, stride=1, act_fn=nn.SiLU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.gn1 = MyGroupNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,
                               padding=1, bias=False)
        self.gn2 = MyGroupNorm(out_channels)
        self.act = act_fn()

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                MyGroupNorm(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        y = self.act(self.gn1(self.conv1(x)))
        y = self.gn2(self.conv2(y))
        return self.act(residual + y)


class ResNetV1Encoder(nn.Module):
    """ResNetV1-34-Bridge encoder.

    Architecture (matching jaxrl_m exactly):
        Input: uint8 images -> [-1, 1] float
        + spatial coordinates (2 channels)
        Conv7x7/2 -> GN -> act -> MaxPool3x3/2
        Stage 1: 3 blocks, 64 filters
        Stage 2: 4 blocks, 128 filters (stride 2 at first block)
        Stage 3: 6 blocks, 256 filters (stride 2 at first block)
        Stage 4: 3 blocks, 512 filters (stride 2 at first block)
        Average pooling -> 512-dim
    """

    def __init__(self, in_channels=6, add_spatial_coordinates=True,
                 stage_sizes=(3, 4, 6, 3), num_filters=64):
        super().__init__()
        self.add_spatial_coordinates = add_spatial_coordinates
        if add_spatial_coordinates:
            self.spatial_coords = AddSpatialCoordinates()
            actual_in = in_channels + 2
        else:
            self.spatial_coords = None
            actual_in = in_channels

        # Initial conv: 7x7, stride 2, padding 3 (matching JAX padding=[(3,3),(3,3)])
        self.conv_init = nn.Conv2d(actual_in, num_filters, 7, stride=2,
                                   padding=3, bias=False)
        self.gn_init = MyGroupNorm(num_filters)
        self.act = nn.SiLU()

        # Build residual stages
        self.stages = nn.ModuleList()
        in_ch = num_filters
        for i, num_blocks in enumerate(stage_sizes):
            blocks = []
            out_ch = num_filters * (2 ** i)
            for j in range(num_blocks):
                stride = 2 if i > 0 and j == 0 else 1
                blocks.append(ResNetBlock(in_ch, out_ch, stride=stride, act_fn=nn.SiLU))
                in_ch = out_ch
            self.stages.append(nn.Sequential(*blocks))

        self.output_dim = in_ch  # 512

        self._init_weights()

    def _init_weights(self):
        """Initialize weights matching JAX defaults.

        JAX Conv uses kaiming_normal, Dense uses xavier_uniform.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, H, W) float tensor, already in [-1, 1] range.
        """
        if self.add_spatial_coordinates:
            x = self.spatial_coords(x)

        x = self.conv_init(x)
        x = self.gn_init(x)
        x = self.act(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        for stage in self.stages:
            x = stage(x)

        # Average pool over spatial dims
        x = x.mean(dim=(-2, -1))  # (B, 512)
        return x


class GCBCPolicy(nn.Module):
    """Goal-Conditioned Behavior Cloning policy.

    Matches the JAX GCBCAgent architecture:
        1. Normalize images: uint8 / 127.5 - 1.0
        2. Early goal concat: cat(obs, goal) on channel dim -> 6ch
        3. ResNet encoder -> 512-dim
        4. Optional proprio concat -> 512 + proprio_dim
        5. MLP (256, 256, 256) with SiLU + dropout 0.1 + activate_final=True
        6. Linear -> action_dim (means)
        7. Fixed std = 1.0
    """

    def __init__(self, action_dim=23, use_proprio=False, proprio_dim=23,
                 hidden_dims=(256, 256, 256), dropout_rate=0.1):
        super().__init__()
        self.action_dim = action_dim
        self.use_proprio = use_proprio

        # Vision encoder: 6 input channels (obs RGB + goal RGB)
        self.encoder = ResNetV1Encoder(
            in_channels=6,
            add_spatial_coordinates=True,
        )

        # MLP policy head
        encoder_out_dim = self.encoder.output_dim  # 512
        if use_proprio:
            encoder_out_dim += proprio_dim

        layers = []
        in_dim = encoder_out_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.SiLU())
            in_dim = h_dim
        self.mlp = nn.Sequential(*layers)

        # Action output head
        self.action_head = nn.Linear(in_dim, action_dim)

        # Fixed log_std = log(1.0) = 0.0
        self.register_buffer("log_std", torch.zeros(action_dim))

        self._init_mlp_weights()

    def _init_mlp_weights(self):
        """Initialize MLP with xavier_uniform (matching JAX default_init)."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, obs_image, goal_image, proprio=None, train=True):
        """Forward pass.

        Args:
            obs_image: (B, H, W, 3) uint8 tensor (channels-last, matching JAX)
            goal_image: (B, H, W, 3) uint8 tensor
            proprio: (B, proprio_dim) float tensor, optional
            train: bool, enables dropout

        Returns:
            means: (B, action_dim) action means
            log_stds: (B, action_dim) log standard deviations (fixed at 0)
        """
        # Normalize to [-1, 1] matching JAX: x / 127.5 - 1.0
        obs = obs_image.float() / 127.5 - 1.0
        goal = goal_image.float() / 127.5 - 1.0

        # Convert NHWC -> NCHW for PyTorch conv
        obs = obs.permute(0, 3, 1, 2)
        goal = goal.permute(0, 3, 1, 2)

        # Early goal concat on channel dim: (B, 6, H, W)
        x = torch.cat([obs, goal], dim=1)

        # Encode
        encoding = self.encoder(x)  # (B, 512)

        # Concat proprio
        if self.use_proprio and proprio is not None:
            encoding = torch.cat([encoding, proprio], dim=-1)

        # MLP
        if not train:
            # Disable dropout at eval
            self.mlp.eval()
        else:
            self.mlp.train()
        features = self.mlp(encoding)

        # Action means
        means = self.action_head(features)

        # Fixed std broadcast
        log_stds = self.log_std.expand_as(means)

        return means, log_stds

    def get_action(self, obs_image, goal_image, proprio=None, argmax=True):
        """Get action for inference (no grad, eval mode).

        Args:
            obs_image: (B, H, W, 3) uint8
            goal_image: (B, H, W, 3) uint8
            proprio: optional (B, D)
            argmax: if True, return mean; else sample

        Returns:
            actions: (B, action_dim)
        """
        with torch.no_grad():
            means, log_stds = self.forward(obs_image, goal_image, proprio, train=False)
            if argmax:
                return means
            else:
                stds = torch.exp(log_stds)
                return means + stds * torch.randn_like(means)

    def compute_loss(self, obs_image, goal_image, actions, proprio=None):
        """Compute NLL loss matching JAX GCBCAgent.

        Loss = -log_prob(actions) under N(means, std=1)
             = 0.5 * sum((actions - means)^2) + const  (per sample)

        Returns:
            loss: scalar
            metrics: dict with actor_loss, mse, log_probs, etc.
        """
        means, log_stds = self.forward(obs_image, goal_image, proprio, train=True)
        stds = torch.exp(log_stds)  # all 1.0

        # Log probability under MultivariateNormalDiag
        # log_prob = -0.5 * sum((a - mu)^2 / sigma^2) - sum(log(sigma)) - 0.5*D*log(2*pi)
        diff = actions - means
        log_probs = -0.5 * (diff / stds).pow(2).sum(dim=-1) \
                    - log_stds.sum(dim=-1) \
                    - 0.5 * self.action_dim * math.log(2 * math.pi)

        actor_loss = -log_probs.mean()
        mse = (diff.pow(2)).sum(dim=-1).mean()

        metrics = {
            "actor_loss": actor_loss.item(),
            "mse": mse.item(),
            "log_probs": log_probs.mean().item(),
            "pi_actions": means.mean().item(),
            "mean_std": stds.mean().item(),
            "max_std": stds.max().item(),
        }

        return actor_loss, metrics
