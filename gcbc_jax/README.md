# Goal-Conditioned BC (JAX) for ISpatialGym

Train the original bridge_data_v2 Goal-Conditioned BC agent on ISpatialGym task demonstrations. Uses the unmodified `GCBCAgent` with `resnetv1-34-bridge` encoder in JAX/Flax.

## 1. Install

Run `setup.sh` from the `bridge_data_v2` directory:

```bash
cd il/bridge_data_v2
chmod +x setup.sh
./setup.sh
```

This creates a conda environment called `jaxrl` with Python 3.10, JAX 0.4.13 (CUDA 12), TensorFlow 2.13, Flax 0.7, and all required dependencies.

Activate it:

```bash
conda activate jaxrl
```

### Requirements

- NVIDIA GPU with CUDA 12 compatible driver (tested with RTX 4090, driver 580.x)
- Miniconda or Miniforge
- ~5 GB disk for the conda env

### Troubleshooting

| Error | Fix |
|---|---|
| `jaxlib version X is newer than and incompatible with jax version 0.4.13` | Run `pip install "jaxlib==0.4.13+cuda12.cudnn89" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html` |
| `AttributeError: module 'scipy.linalg' has no attribute 'tril'` | Run `pip install "scipy==1.11.4"` |
| `no GPU/TPU found` | Check `nvidia-smi`, ensure CUDA driver is installed |

## 2. Prepare Data

The training pipeline expects data in TFRecord format. The conversion script reads ISpatialGym parquet files (low-dim state + actions) and MP4 video files (head camera RGB), and writes TFRecords that the original `BridgeDataset` can load.

### Source data layout

```
datasets/ispatialgym-eval-demos/
    data/task-XXXX/
        episode_XXXX.parquet          # Low-dim: state, action, goal image path
    videos/task-XXXX/
        observation.images.rgb.head/
            episode_XXXX.mp4          # Head camera RGB video (frame-aligned with parquet)
```

Each parquet file has an `image_condition_path` column pointing to the goal reference image (PNG).

### Convert to TFRecords

From the `bridge_data_v2` directory:

```bash
conda activate jaxrl

python -m gcbc_jax.convert_to_tfrecord \
    --data_dir /path/to/datasets/ispatialgym-eval-demos/data/task-0053 \
    --output_dir gcbc_jax/tfrecords/task-0053 \
    --project_root /path/to/behavior-1k-private \
    --image_size 256
```

**Arguments:**

| Argument | Description |
|---|---|
| `--data_dir` | Directory containing parquet episode files |
| `--output_dir` | Where to write TFRecords (creates `train/` and `val/` subdirs) |
| `--project_root` | Root for resolving relative `image_condition_path` in parquet (some tasks use relative paths) |
| `--video_dir` | Video directory (auto-detected from `data_dir` if not set: `../videos/task-XXXX`) |
| `--image_size` | Resize all images to this resolution (default: 256, matching bridge_data_v2) |
| `--val_ratio` | Fraction of episodes for validation (default: 0.2) |
| `--seed` | Random seed for train/val split (default: 42) |

**Output:**

```
gcbc_jax/tfrecords/task-0053/
    train/
        episode_XXXX.tfrecord    # One trajectory per file
        ...
    val/
        episode_XXXX.tfrecord
        ...
    action_proprio_metadata.json  # Action normalization stats (mean/std)
```

### Filtering specific episodes

If you only want a subset of episodes (e.g., the 10 from `ispatialgym-fixed-goal-instances`), create a directory with symlinks first:

```bash
mkdir -p /tmp/my_episodes
for ep in 0053000000000000 0053000000001000 ...; do
    ln -s /path/to/data/task-0053/episode_${ep}.parquet /tmp/my_episodes/
done

python -m gcbc_jax.convert_to_tfrecord \
    --data_dir /tmp/my_episodes \
    --video_dir /path/to/videos/task-0053 \
    --output_dir gcbc_jax/tfrecords/task-0053 \
    --project_root /path/to/behavior-1k-private
```

## 3. Train

From the `bridge_data_v2` directory:

```bash
conda activate jaxrl

python -m gcbc_jax.train \
    --tfrecord_dir gcbc_jax/tfrecords/task-0053 \
    --save_dir outputs/gcbc_jax_task0053 \
    --num_steps 50000 \
    --batch_size 256 \
    --use_wandb \
    --wandb_project gcbc-ispatialgym \
    --run_name my_run
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--tfrecord_dir` | (required) | Directory with `train/`, `val/`, and `action_proprio_metadata.json` |
| `--save_dir` | `outputs/gcbc_jax` | Checkpoint output directory |
| `--encoder` | `resnetv1-34-bridge` | Vision encoder (from bridge_data_v2 encoder registry) |
| `--num_steps` | 100000 | Total training steps |
| `--batch_size` | 256 | Batch size |
| `--lr` | 3e-4 | Peak learning rate |
| `--warmup_steps` | 2000 | Linear LR warmup steps |
| `--augment` / `--no_augment` | augment on | Image augmentation (crop, color jitter) |
| `--log_interval` | 100 | Log training metrics every N steps |
| `--eval_interval` | 5000 | Run validation every N steps |
| `--save_interval` | 5000 | Save checkpoint every N steps |
| `--use_wandb` | off | Enable WandB logging |
| `--wandb_project` | `gcbc-ispatialgym` | WandB project name |
| `--run_name` | `gcbc_jax` | WandB run name |
| `--seed` | 42 | Random seed |

**Outputs:**

```
outputs/gcbc_jax_task0053/
    checkpoint_5000                    # Flax checkpoint at step 5000
    checkpoint_10000
    ...
    action_proprio_metadata.json       # Copied from tfrecord_dir for eval
```

### WandB metrics

| Key | Description |
|---|---|
| `training/actor_loss` | Gaussian NLL loss (equivalent to MSE with fixed std=1) |
| `training/mse` | Mean squared error between predicted and target actions |
| `training/lr` | Current learning rate |
| `validation/actor_loss` | Validation NLL loss |
| `validation/mse` | Validation MSE |

### What happens each training step

#### 1. Batch sampling

A batch of `B` transitions is sampled from the TFRecord train set via `tf.data`. Each element in the batch contains:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `observations/image` | `(B, 256, 256, 3)` | uint8 | Current head camera frame |
| `goals/image` | `(B, 256, 256, 3)` | uint8 | Fixed goal reference image (same for all timesteps in an episode) |
| `actions` | `(B, 23)` | float32 | Z-score normalized target action |
| `observations/proprio` | `(B, 23)` | float32 | Dummy zeros (proprioception disabled) |
| `goals/proprio` | `(B, 23)` | float32 | Dummy zeros |
| `masks` | `(B,)` | bool | Inverse of `terminals` |

The `FixedGoalBridgeDataset` overrides standard goal relabeling: instead of sampling a future observation as the goal (as in the original bridge_data_v2), it uses the fixed reference image stored in each TFRecord. Actions are also not relabeled since they are already ground-truth motor commands (not deltas from proprioception).

Image augmentation (if enabled) is applied to both observation and goal images identically: random resized crop (scale 0.8–1.0, ratio 0.9–1.1), brightness (±0.2), contrast (0.8–1.2), saturation (0.8–1.2), hue (±0.1).

#### 2. Forward pass

```
obs_image (B, 256, 256, 3) uint8     goal_image (B, 256, 256, 3) uint8
           \                                /
            +----- concat along channel ----+
                          |
              (B, 256, 256, 6) float32 (after /255 normalization)
                          |
                  +-- add spatial coords --+
                  |  (2 extra channels:    |
                  |   x ∈ [-1,1],          |
                  |   y ∈ [-1,1])          |
                          |
              (B, 256, 256, 8) float32
                          |
              ResNetV1-34-Bridge encoder
              (GroupNorm(4), Swish activation,
               avg spatial pooling → 512-dim)
                          |
                   (B, 512) float32
                          |
                   MLP policy head
              Dense(512→256) + SiLU + Dropout(0.1)
              Dense(256→256) + SiLU + Dropout(0.1)
              Dense(256→256) + SiLU + Dropout(0.1)
              Dense(256→23)  → action means
                          |
                   (B, 23) float32
                          |
         MultivariateNormalDiag(loc=means, scale=1.0)
                          |
                 π(a | obs, goal)
```

**Key details:**
- `early_goal_concat=True`: obs and goal images are concatenated along the channel axis *before* encoding, forming a 6-channel input to a single shared ResNet. This is cheaper and more parameter-efficient than encoding each image separately.
- Spatial coordinates (2 extra channels with normalized x,y grid) are prepended, making 8 input channels total.
- The first conv layer of ResNet-34 is modified to accept 8 channels instead of the standard 3.
- `use_proprio=False`: proprioception is not concatenated to the encoder output.
- `fixed_std=[1.0]*23`: the Gaussian policy has a fixed standard deviation of 1.0 for all action dimensions (not learned).

#### 3. Loss computation

The loss is **negative log-likelihood** (NLL) of the ground-truth actions under the predicted Gaussian:

```
log_prob = π.log_prob(a_target)          # scalar per sample
actor_loss = -mean(log_prob)             # averaged over batch
```

With fixed σ=1, the Gaussian NLL simplifies to:

```
-log P(a | μ, σ=1) = 0.5 * ||a - μ||² + const
```

So `actor_loss ≈ 0.5 * MSE + const`. The constant (involving `log(2π)` and action dim) does not affect gradients. Minimizing NLL with fixed std is equivalent to minimizing MSE.

#### 4. Gradient update

- **Optimizer:** Adam with warmup + cosine decay schedule
- **Warmup:** Linear ramp from 0 to `lr` over the first 2000 steps
- **Decay:** Cosine annealing from `lr` to 0 over the remaining steps
- **Gradient computation:** `jax.value_and_grad` on the actor loss
- **Parameter update:** Single Adam step on all parameters (encoder + MLP head)

The agent's `update()` method calls `state.apply_loss_fns()` which computes the loss + gradient and applies the optimizer in one step. The updated agent state (parameters + optimizer state) is returned.

#### 5. Metrics logged

| Metric | Description |
|---|---|
| `actor_loss` | NLL loss (primary training objective) |
| `mse` | Mean squared error `mean(Σ_d (μ_d - a_d)²)` (for monitoring, not used in gradient) |
| `log_probs` | Mean log probability of target actions (higher = better fit) |
| `pi_actions` | Mean of predicted action means (sanity check) |
| `mean_std` | Mean policy std across action dims (fixed at 1.0 here) |
| `max_std` | Max policy std across action dims (fixed at 1.0 here) |
| `lr` | Current learning rate (ramps up then decays) |

### Architecture summary

- **Encoder:** ResNetV1-34-Bridge (GroupNorm, Swish, spatial coordinates, avg pool)
- **Goal fusion:** Early concatenation (obs + goal images along channel dim → 6-channel input)
- **Policy head:** MLP (256, 256, 256) → Gaussian with fixed std=1
- **Loss:** Negative log-likelihood (= MSE + constant when std is fixed)
- **Observations:** Head camera RGB only, resized to 256x256
- **Proprioception:** Disabled (`use_proprio=False`)
- **Parameters:** ~21.6M

See `goal-conditioned BC.md` for a detailed architecture description.

### GPU memory

| Batch size | Image size | Approx. GPU memory |
|---|---|---|
| 256 | 256x256 | ~18 GB |
| 128 | 256x256 | ~10 GB |
| 32 | 256x256 | ~4 GB |

## 4. Evaluate

See `gcbc_jax/eval_policy.py` for the JAX→PyTorch wrapper that interfaces with `eval_ispatialgym.py`.

```bash
# From the OmniGibson directory, using the behavior_dev conda env:
python -m omnigibson.learning.eval_ispatialgym \
    --task-name camera_relocalization \
    --episode-id <episode_id> \
    --policy-checkpoint /path/to/outputs/gcbc_jax_task0053 \
    --headless
```
