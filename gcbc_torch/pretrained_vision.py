"""Pretrained vision encoder wrappers for GCBC late-fusion baseline.

Supports:
  - DINOv2 (facebook/dinov2-base) via Dinov2Model
  - SigLIP (google/siglip-base-patch16-224) via SiglipVisionModel
  - DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m) via DINOv3ViTModel
    (requires transformers >= 4.56.0)

All use Hugging Face Transformers vision-only classes with lazy imports
so that the resnet path still works without transformers installed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default HF model IDs for each encoder choice
ENCODER_HF_DEFAULTS = {
    "dinov2-base": "facebook/dinov2-base",
    "siglip-base": "google/siglip-base-patch16-224",
    "dinov3-vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}

# Preprocessing constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


def _ensure_transformers(min_version=None):
    """Raise a clear error if transformers is not installed or too old."""
    try:
        import transformers
    except ImportError:
        raise ImportError(
            "Pretrained vision encoders require the `transformers` and `safetensors` "
            "packages. Install with:\n"
            "  pip install -U 'transformers>=4.56.0' safetensors"
        ) from None
    if min_version is not None:
        from packaging.version import Version
        if Version(transformers.__version__) < Version(min_version):
            raise ImportError(
                f"DINOv3 requires transformers >= {min_version}, "
                f"but found {transformers.__version__}. Upgrade with:\n"
                f"  pip install -U 'transformers>={min_version}'"
            )


class PretrainedVisionEncoder(nn.Module):
    """Shared pretrained vision encoder for GCBC late-fusion.

    Takes NHWC uint8 input, preprocesses to the encoder's expected tensor
    format using batched torch ops, and returns one feature vector per image.

    Args:
        encoder_type: "dinov2-base", "siglip-base", or "dinov3-vitl16"
        model_name_or_path: HF repo id or local directory. None uses default.
        freeze: If True, freeze encoder weights and keep in eval mode.
        load_pretrained_weights: If True, load from pretrained via
            .from_pretrained(). If False, instantiate from config only
            (for tests without internet or checkpoint resume).
        encoder_config_dict: If provided and load_pretrained_weights=False,
            reconstruct the encoder config from this dict (saved in checkpoint)
            instead of downloading from HF. Enables fully offline resume.
    """

    def __init__(self, encoder_type, model_name_or_path=None, freeze=True,
                 load_pretrained_weights=True, encoder_config_dict=None):
        super().__init__()
        min_ver = "4.56.0" if encoder_type == "dinov3-vitl16" else None
        _ensure_transformers(min_version=min_ver)

        self.encoder_type = encoder_type
        self.freeze = freeze

        model_id = model_name_or_path or ENCODER_HF_DEFAULTS[encoder_type]

        if encoder_type == "dinov2-base":
            from transformers import Dinov2Config, Dinov2Model

            if load_pretrained_weights:
                self.backbone = Dinov2Model.from_pretrained(model_id)
            elif encoder_config_dict is not None:
                config = Dinov2Config(**encoder_config_dict)
                self.backbone = Dinov2Model(config)
            else:
                config = Dinov2Config.from_pretrained(model_id)
                self.backbone = Dinov2Model(config)
            self.output_dim = self.backbone.config.hidden_size  # 768

            self.register_buffer(
                "pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer(
                "pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        elif encoder_type == "siglip-base":
            from transformers import SiglipVisionConfig, SiglipVisionModel

            if load_pretrained_weights:
                self.backbone = SiglipVisionModel.from_pretrained(model_id)
            elif encoder_config_dict is not None:
                config = SiglipVisionConfig(**encoder_config_dict)
                self.backbone = SiglipVisionModel(config)
            else:
                config = SiglipVisionConfig.from_pretrained(model_id)
                self.backbone = SiglipVisionModel(config)
            self.output_dim = self.backbone.config.hidden_size  # 768

            self.register_buffer(
                "pixel_mean", torch.tensor(SIGLIP_MEAN).view(1, 3, 1, 1))
            self.register_buffer(
                "pixel_std", torch.tensor(SIGLIP_STD).view(1, 3, 1, 1))

        elif encoder_type == "dinov3-vitl16":
            from transformers import DINOv3ViTConfig, DINOv3ViTModel

            if load_pretrained_weights:
                self.backbone = DINOv3ViTModel.from_pretrained(model_id)
            elif encoder_config_dict is not None:
                config = DINOv3ViTConfig.from_dict(encoder_config_dict)
                self.backbone = DINOv3ViTModel(config)
            else:
                config = DINOv3ViTConfig.from_pretrained(model_id)
                self.backbone = DINOv3ViTModel(config)
            self.output_dim = self.backbone.config.hidden_size  # 1024

            self.register_buffer(
                "pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer(
                "pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        else:
            raise ValueError(
                f"Unknown encoder type: {encoder_type!r}. "
                f"Supported: {list(ENCODER_HF_DEFAULTS)}")

        if freeze:
            self._freeze()

    def _freeze(self):
        """Freeze all backbone parameters and set to eval mode."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def _preprocess_dinov2(self, x):
        """Batched DINOv2 preprocessing on NCHW float tensors.

        Pipeline: uint8 [0,255] -> float [0,1] -> resize shortest edge 256
        -> center crop 224x224 -> ImageNet normalize.
        """
        x = x.permute(0, 3, 1, 2).float() / 255.0

        # Resize shortest edge to 256, preserving aspect ratio
        _, _, h, w = x.shape
        if h <= w:
            new_h = 256
            new_w = int(round(w * 256 / h))
        else:
            new_w = 256
            new_h = int(round(h * 256 / w))
        if new_h != h or new_w != w:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear",
                              align_corners=False)

        # Center crop to 224x224
        start_h = (new_h - 224) // 2
        start_w = (new_w - 224) // 2
        x = x[:, :, start_h:start_h + 224, start_w:start_w + 224]

        x = (x - self.pixel_mean) / self.pixel_std
        return x

    def _preprocess_siglip(self, x):
        """Batched SigLIP preprocessing on NCHW float tensors.

        Pipeline: uint8 [0,255] -> float [0,1] -> resize 224x224
        -> normalize mean=0.5, std=0.5.
        """
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bilinear",
                          align_corners=False)
        x = (x - self.pixel_mean) / self.pixel_std
        return x

    def _preprocess_dinov3(self, x):
        """Batched DINOv3 preprocessing on NCHW float tensors.

        Pipeline: uint8 [0,255] -> float [0,1] -> resize 224x224
        -> ImageNet normalize.
        """
        x = x.permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bilinear",
                          align_corners=False)
        x = (x - self.pixel_mean) / self.pixel_std
        return x

    def forward(self, x):
        """Encode a batch of images to feature vectors.

        Args:
            x: (B, H, W, 3) uint8 tensor (NHWC, matching dataset convention).

        Returns:
            (B, output_dim) float tensor.
        """
        if self.encoder_type == "dinov2-base":
            x = self._preprocess_dinov2(x)
        elif self.encoder_type == "dinov3-vitl16":
            x = self._preprocess_dinov3(x)
        else:
            x = self._preprocess_siglip(x)

        outputs = self.backbone(pixel_values=x)

        if self.encoder_type == "dinov2-base":
            # CLS token from the last hidden state
            return outputs.last_hidden_state[:, 0, :]
        elif self.encoder_type == "dinov3-vitl16":
            # Pooler output if available, else CLS token
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            return outputs.last_hidden_state[:, 0, :]
        else:
            # SigLIP: multi-head attention pooler output
            return outputs.pooler_output

    def train(self, mode=True):
        """Override to keep frozen backbone in eval mode."""
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self
