"""PyTorch Goal-Conditioned IQL (Implicit Q-Learning) policy.

Port of jaxrl_m GCIQLAgent with the same encoder/MLP architecture as GCBCPolicy:
  - Shared ResNetV1-34 encoder with early goal concatenation
  - Actor: MLP → MultivariateNormal (fixed std=1.0)
  - Value: MLP → scalar V(s,g)
  - Critic: MLP(cat(encoding, action)) → scalar Q(s,g,a)
  - Expectile value loss, advantage-weighted actor loss, MSE critic loss
  - Polyak-averaged target network for value function
  - Negative goal sampling within batch
"""

import math

import torch
import torch.nn as nn

from .model import ResNetV1Encoder


def expectile_loss(diff, expectile=0.7):
    """Asymmetric squared loss: penalizes underestimation more."""
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return weight * diff.pow(2)


class GCIQLPolicy(nn.Module):
    """Goal-Conditioned IQL with shared encoder, three heads."""

    def __init__(self, action_dim=23, use_proprio=False, proprio_dim=23,
                 hidden_dims=(256, 256, 256), dropout_rate=0.1,
                 discount=0.98, expectile=0.7, temperature=1.0,
                 negative_proportion=0.1, adv_clip_max=100.0):
        super().__init__()
        self.action_dim = action_dim
        self.use_proprio = use_proprio
        self.discount = discount
        self.expectile = expectile
        self.temperature = temperature
        self.negative_proportion = negative_proportion
        self.adv_clip_max = adv_clip_max

        # Shared vision encoder (obs + goal early concat → 6ch)
        self.encoder = ResNetV1Encoder(
            in_channels=6,
            add_spatial_coordinates=True,
        )

        encoder_out_dim = self.encoder.output_dim  # 512
        if use_proprio:
            encoder_out_dim += proprio_dim

        # Actor MLP + action head
        self.actor_mlp = self._build_mlp(encoder_out_dim, hidden_dims, dropout_rate)
        self.actor_head = nn.Linear(hidden_dims[-1], action_dim)
        self.register_buffer("log_std", torch.zeros(action_dim))

        # Value MLP + head
        self.value_mlp = self._build_mlp(encoder_out_dim, hidden_dims, dropout_rate)
        self.value_head = nn.Linear(hidden_dims[-1], 1)

        # Critic MLP + head (input includes action)
        self.critic_mlp = self._build_mlp(
            encoder_out_dim + action_dim, hidden_dims, dropout_rate)
        self.critic_head = nn.Linear(hidden_dims[-1], 1)

        self._init_mlp_weights()

    @staticmethod
    def _build_mlp(in_dim, hidden_dims, dropout_rate):
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(d, h), nn.Dropout(dropout_rate), nn.SiLU()])
            d = h
        return nn.Sequential(*layers)

    def _init_mlp_weights(self):
        for module in [self.actor_mlp, self.value_mlp, self.critic_mlp]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        for head in [self.actor_head, self.value_head, self.critic_head]:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def _encode(self, obs_image, goal_image, proprio=None):
        """Shared encoder: normalize, concat, ResNet, optional proprio."""
        obs = obs_image.float() / 127.5 - 1.0
        goal = goal_image.float() / 127.5 - 1.0
        obs = obs.permute(0, 3, 1, 2)
        goal = goal.permute(0, 3, 1, 2)
        x = torch.cat([obs, goal], dim=1)  # (B, 6, H, W)
        encoding = self.encoder(x)  # (B, 512)
        if self.use_proprio and proprio is not None:
            encoding = torch.cat([encoding, proprio], dim=-1)
        return encoding

    def forward_actor(self, encoding, train=True):
        if not train:
            self.actor_mlp.eval()
        else:
            self.actor_mlp.train()
        features = self.actor_mlp(encoding)
        means = self.actor_head(features)
        return means, self.log_std.expand_as(means)

    def forward_value(self, encoding, train=True):
        if not train:
            self.value_mlp.eval()
        else:
            self.value_mlp.train()
        features = self.value_mlp(encoding)
        return self.value_head(features).squeeze(-1)  # (B,)

    def forward_critic(self, encoding, actions, train=True):
        if not train:
            self.critic_mlp.eval()
        else:
            self.critic_mlp.train()
        x = torch.cat([encoding, actions], dim=-1)
        features = self.critic_mlp(x)
        return self.critic_head(features).squeeze(-1)  # (B,)

    def compute_loss(self, batch, target_model):
        """Compute IQL losses (critic, value, actor).

        Args:
            batch: dict with obs_image, goal_image, next_obs_image, actions,
                   rewards, masks, and optionally obs_proprio, next_obs_proprio.
            target_model: EMA copy of this model for target V computation.

        Returns:
            total_loss: scalar (sum of three losses)
            metrics: dict
        """
        obs_image = batch["obs_image"]
        goal_image = batch["goal_image"]
        next_obs_image = batch["next_obs_image"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        masks = batch["masks"]
        proprio = batch.get("obs_proprio") if self.use_proprio else None
        next_proprio = batch.get("next_obs_proprio") if self.use_proprio else None

        # Negative goal sampling: roll goals within batch
        B = obs_image.shape[0]
        if self.negative_proportion > 0:
            neg_mask = torch.rand(B, device=obs_image.device) < self.negative_proportion
            neg_indices = torch.roll(torch.arange(B, device=obs_image.device), -1)
            orig_indices = torch.arange(B, device=obs_image.device)
            indices = torch.where(neg_mask, neg_indices, orig_indices)
            goal_image = goal_image[indices]
            rewards = torch.where(neg_mask.float() > 0, -1.0, rewards)

        # Encode current observations
        encoding = self._encode(obs_image, goal_image, proprio)

        # --- Target V(s', g) using EMA model ---
        with torch.no_grad():
            next_encoding_target = target_model._encode(
                next_obs_image, goal_image, next_proprio)
            next_v = target_model.forward_value(next_encoding_target, train=False)

        # --- Critic loss: MSE(Q, r + γ * V_target * mask) ---
        target_q = rewards + self.discount * next_v * masks
        q = self.forward_critic(encoding, actions, train=True)
        critic_loss = (q - target_q.detach()).pow(2).mean()

        # --- Value loss: expectile_loss(Q.detach() - V) ---
        with torch.no_grad():
            q_detached = self.forward_critic(encoding.detach(), actions, train=False)
        v = self.forward_value(encoding, train=True)
        value_diff = q_detached - v
        value_loss = expectile_loss(value_diff, self.expectile).mean()

        # --- Actor loss: -exp(advantage / temp) * log_prob(a) ---
        with torch.no_grad():
            adv = target_q - v.detach()
            exp_adv = torch.exp(adv / self.temperature)
            exp_adv = torch.clamp(exp_adv, max=self.adv_clip_max)

        means, log_stds = self.forward_actor(encoding, train=True)
        stds = torch.exp(log_stds)
        diff = actions - means
        log_probs = (-0.5 * (diff / stds).pow(2).sum(dim=-1)
                     - log_stds.sum(dim=-1)
                     - 0.5 * self.action_dim * math.log(2 * math.pi))
        actor_loss = -(exp_adv * log_probs).mean()

        total_loss = critic_loss + value_loss + actor_loss

        mse = (actions - means).pow(2).sum(dim=-1).mean()
        metrics = {
            "critic_loss": critic_loss.item(),
            "value_loss": value_loss.item(),
            "actor_loss": actor_loss.item(),
            "total_loss": total_loss.item(),
            "mse": mse.item(),
            "q": q.mean().item(),
            "v": v.mean().item(),
            "target_q": target_q.mean().item(),
            "advantage": adv.mean().item(),
            "log_probs": log_probs.mean().item(),
        }

        return total_loss, metrics

    def get_action(self, obs_image, goal_image, proprio=None, argmax=True,
                   target_state_dict=None):
        """Get action for inference (actor head only)."""
        with torch.no_grad():
            encoding = self._encode(obs_image, goal_image, proprio)
            means, log_stds = self.forward_actor(encoding, train=False)
            if argmax:
                return means
            stds = torch.exp(log_stds)
            return means + stds * torch.randn_like(means)
