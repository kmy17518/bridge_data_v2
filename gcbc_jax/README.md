# Goal-Conditioned BC (JAX) for ISpatialGym

Train the original bridge_data_v2 Goal-Conditioned BC agent on ISpatialGym task demonstrations. Uses the unmodified `GCBCAgent` with `resnetv1-34-bridge` encoder in JAX/Flax.

## 1. Install

### Option A: Standalone JAX environment

Run `setup.sh` from the `bridge_data_v2` directory:

```bash
cd il/bridge_data_v2
chmod +x setup.sh
./setup.sh
```

This creates a conda environment called `jaxrl` with Python 3.10, JAX 0.4.13 (CUDA 12), TensorFlow 2.13, Flax 0.7, and all required dependencies.

```bash
conda activate jaxrl
```

### Option B: Add to existing behavior environment

From the project root, run the main setup script with `--train`:

```bash
./setup.sh --train
```

This installs JAX/Flax training dependencies into the existing `behavior` conda env, enabling both training and evaluation from a single environment.

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
datasets-final/ispatialgym-demos/
    data/task-XXXX/
        episode_XXXX.parquet          # Low-dim: state (256), action (23), goal image path
    videos/task-XXXX/
        observation.images.rgb.head/
            episode_XXXX.mp4          # Head camera RGB video (frame-aligned with parquet)
```

Each parquet file has an `image_condition_path` column pointing to the goal reference image (PNG).

### Convert to TFRecords

From the `bridge_data_v2` directory:

```bash
# From il/bridge_data_v2, with PROJECT_ROOT pointing to behavior-1k-private:
PROJECT_ROOT=$(cd ../../ && pwd)

python -m gcbc_jax.convert_to_tfrecord \
    --data_dir $PROJECT_ROOT/datasets-final/ispatialgym-demos/data/task-0053 \
    --output_dir gcbc_jax/tfrecords/task-0053-final \
    --project_root $PROJECT_ROOT \
    --image_size 256
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | (required) | Directory containing parquet episode files |
| `--output_dir` | (required) | Where to write TFRecords (creates `train/`, `val/`, `test/` subdirs) |
| `--project_root` | None | Root for resolving relative `image_condition_path` in parquet |
| `--video_dir` | auto | Video directory (auto-detected from `data_dir`: `../videos/task-XXXX`) |
| `--image_size` | 256 | Resize all images to this resolution |
| `--train_ratio` | 0.8 | Fraction of episodes for training |
| `--val_ratio` | 0.1 | Fraction of episodes for validation |
| `--test_ratio` | 0.1 | Fraction of episodes for testing |
| `--seed` | 42 | Random seed for split |

**What gets stored:**

- **Observation images:** Head camera RGB frames, resized to `image_size` (uint8)
- **Proprioception:** Full 256-dim `observation.state` from parquet (float32)
- **Actions:** 23-dim raw motor commands (float32)
- **Goal image:** Reference image from `image_condition_path`, resized to `image_size` (uint8)
- **Normalization stats:** `action_proprio_metadata.json` with per-dim mean/std for both actions and proprio, computed from training set only

**Output:**

```
gcbc_jax/tfrecords/task-0053-final/
    train/
        episode_XXXX.tfrecord    # One trajectory per file
    val/
        episode_XXXX.tfrecord
    test/
        episode_XXXX.tfrecord
    action_proprio_metadata.json  # Action + proprio normalization stats (mean/std)
```

## 3. Train

From the `bridge_data_v2` directory. Two modes are available: **step mode** (default) and **epoch mode**.

### Step mode

```bash
python -m gcbc_jax.train \
    --tfrecord_dir gcbc_jax/tfrecords/task-0053-final \
    --save_dir outputs/gcbc_jax_task0053_proprio \
    --num_steps 50000 \
    --batch_size 256 \
    --use_proprio --normalize_proprio \
    --use_wandb --wandb_project gcbc-ispatialgym --run_name gcbc_proprio_300ep
```

### Epoch mode

```bash
python -m gcbc_jax.train \
    --tfrecord_dir gcbc_jax/tfrecords/task-0053-final \
    --save_dir outputs/gcbc_jax_task0053_proprio \
    --num_epochs 68 \
    --batch_size 256 \
    --use_proprio --normalize_proprio \
    --log_interval 1 --eval_interval 5 --save_interval 5 \
    --use_wandb --wandb_project gcbc-ispatialgym --run_name gcbc_proprio_300ep
```

In epoch mode, `--log_interval`, `--eval_interval`, and `--save_interval` are in epochs.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--tfrecord_dir` | (required) | Directory with `train/`, `val/`, and `action_proprio_metadata.json` |
| `--save_dir` | `outputs/gcbc_jax` | Checkpoint output directory |
| `--encoder` | `resnetv1-34-bridge` | Vision encoder (from bridge_data_v2 encoder registry) |
| `--num_steps` | 100000 | Total training steps (step mode) |
| `--num_epochs` | None | Total training epochs (epoch mode, overrides `--num_steps`) |
| `--batch_size` | 256 | Batch size |
| `--lr` | 3e-4 | Peak learning rate |
| `--warmup_steps` | 2000 | Linear LR warmup steps |
| `--augment` / `--no_augment` | augment on | Image augmentation (crop, color jitter) |
| `--use_proprio` | off | Use 23-dim proprioception (base velocity, trunk, arms, grippers) |
| `--add_eef_proprio` | off | Extend to 37-dim by adding end-effector pos + quat |
| `--normalize_proprio` | off | Normalize proprio to [-1, 1] using JOINT_RANGE bounds |
| `--log_interval` | 100 | Log every N steps (step mode) or N epochs (epoch mode) |
| `--eval_interval` | 5000 / 5 | Eval every N steps/epochs |
| `--save_interval` | 5000 / 5 | Save checkpoint every N steps/epochs |
| `--use_wandb` | off | Enable WandB logging |
| `--wandb_project` | `gcbc-ispatialgym` | WandB project name |
| `--run_name` | `gcbc_jax` | WandB run name |
| `--seed` | 42 | Random seed |

### Proprioception

The full 256-dim `observation.state` is stored in TFRecords. At training time, `--use_proprio` extracts a 23 or 37-dim subset using the same indices as OpenPI / IL_LIB baselines:

**23-dim** (`--use_proprio`):

| Dims | Source | Indices into 256-dim state |
|---|---|---|
| 3 | base velocity | `[253:256]` |
| 4 | trunk joint positions | `[236:240]` |
| 7 | left arm joint positions | `[158:165]` |
| 7 | right arm joint positions | `[197:204]` |
| 1 | left gripper (sum of 2 fingers) | `sum([193:195])` |
| 1 | right gripper (sum of 2 fingers) | `sum([232:234])` |

**37-dim** (`--use_proprio --add_eef_proprio`), extends with:

| Dims | Source | Indices |
|---|---|---|
| 3 | left end-effector position | `[186:189]` |
| 4 | left end-effector quaternion | `[189:193]` |
| 3 | right end-effector position | `[225:228]` |
| 4 | right end-effector quaternion | `[228:232]` |

With `--normalize_proprio`, values are normalized to [-1, 1] using `JOINT_RANGE` and `EEF_POSITION_RANGE` bounds from `OmniGibson/omnigibson/learning/utils/eval_utils.py`. Quaternions are left unnormalized (already in [-1, 1]).

### Outputs

```
outputs/gcbc_jax_task0053_proprio/
    checkpoint_5000                              # Flax msgpack checkpoint at step 5000
    checkpoint_10000
    ...
    checkpoint_50000
    action_proprio_metadata.json                 # Copied from tfrecord_dir for eval
    vis/
        step_5000/
            episode_0053000000097000_actions.png # Predicted vs GT action comparison plots
            episode_0053000000229000_actions.png
            episode_0053000000278000_actions.png
        step_10000/
            ...
```

### WandB metrics

| Key | Description |
|---|---|
| `training/actor_loss` | Gaussian NLL loss (equivalent to MSE with fixed std=1) |
| `training/mse` | Mean squared error between predicted and target actions |
| `training/lr` | Current learning rate |
| `validation/actor_loss` | Validation NLL loss |
| `validation/mse` | Validation MSE |
| `vis/ep{i}_actions` | Predicted vs GT action plots for 3 fixed val trajectories |
| `vis/ep{i}_mse` | Per-episode MSE on fixed val trajectories |

### What happens each training step

#### 1. Batch sampling

A batch of `B` transitions is sampled from the TFRecord train set via `tf.data`. Each element in the batch contains:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `observations/image` | `(B, 256, 256, 3)` | uint8 | Current head camera frame |
| `goals/image` | `(B, 256, 256, 3)` | uint8 | Fixed goal reference image (same for all timesteps in an episode) |
| `actions` | `(B, 23)` | float32 | Z-score normalized target action |
| `observations/proprio` | `(B, D)` | float32 | Proprioception (D=23 or 37 if enabled, else 256 raw) |
| `masks` | `(B,)` | bool | Inverse of `terminals` |

The `FixedGoalBridgeDataset` overrides standard goal relabeling: instead of sampling a future observation as the goal (as in the original bridge_data_v2), it uses the fixed reference image stored in each TFRecord. Actions are also not relabeled since they are already ground-truth motor commands (not deltas from proprioception).

Image augmentation (if enabled) is applied to both observation and goal images identically: random resized crop (scale 0.8-1.0, ratio 0.9-1.1), brightness (+/-0.2), contrast (0.8-1.2), saturation (0.8-1.2), hue (+/-0.1).

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
                  |   x in [-1,1],         |
                  |   y in [-1,1])         |
                          |
              (B, 256, 256, 8) float32
                          |
              ResNetV1-34-Bridge encoder
              (GroupNorm(4), Swish activation,
               avg spatial pooling -> 512-dim)
                          |
                   (B, 512) float32
                          |
              [if use_proprio: concat proprio (B, D)]
                          |
                   (B, 512+D) float32
                          |
                   MLP policy head
              Dense -> SiLU -> Dropout(0.1)  x3
              Dense -> action means
                          |
                   (B, 23) float32
                          |
         MultivariateNormalDiag(loc=means, scale=1.0)
                          |
                 pi(a | obs, goal)
```

**Key details:**
- `early_goal_concat=True`: obs and goal images are concatenated along the channel axis *before* encoding, forming a 6-channel input to a single shared ResNet. This is cheaper and more parameter-efficient than encoding each image separately.
- Spatial coordinates (2 extra channels with normalized x,y grid) are added, making 8 input channels total.
- The first conv layer of ResNet-34 is modified to accept 8 channels instead of the standard 3.
- `use_proprio=True`: proprioception is concatenated to the encoder output before the MLP head. The MLP input is 512+D where D is the proprio dimension (23 or 37).
- `fixed_std=[1.0]*23`: the Gaussian policy has a fixed standard deviation of 1.0 for all action dimensions (not learned).

#### 3. Loss computation

The loss is **negative log-likelihood** (NLL) of the ground-truth actions under the predicted Gaussian:

```
log_prob = pi.log_prob(a_target)          # scalar per sample
actor_loss = -mean(log_prob)             # averaged over batch
```

With fixed sigma=1, the Gaussian NLL simplifies to:

```
-log P(a | mu, sigma=1) = 0.5 * ||a - mu||^2 + const
```

So `actor_loss = 0.5 * MSE + const`. The constant (involving `log(2*pi)` and action dim) does not affect gradients. Minimizing NLL with fixed std is equivalent to minimizing MSE.

#### 4. Gradient update

- **Optimizer:** Adam with warmup + cosine decay schedule
- **Warmup:** Linear ramp from 0 to `lr` over the first 2000 steps
- **Decay:** Cosine annealing from `lr` to 0 over the remaining steps
- **Gradient computation:** `jax.value_and_grad` on the actor loss
- **Parameter update:** Single Adam step on all parameters (encoder + MLP head)

#### 5. Metrics logged

| Metric | Description |
|---|---|
| `actor_loss` | NLL loss (primary training objective) |
| `mse` | Mean squared error `mean(sum_d (mu_d - a_d)^2)` (for monitoring, not used in gradient) |
| `log_probs` | Mean log probability of target actions (higher = better fit) |
| `pi_actions` | Mean of predicted action means (sanity check) |
| `mean_std` | Mean policy std across action dims (fixed at 1.0 here) |
| `max_std` | Max policy std across action dims (fixed at 1.0 here) |
| `lr` | Current learning rate (ramps up then decays) |

### Architecture summary

- **Encoder:** ResNetV1-34-Bridge (GroupNorm, Swish, spatial coordinates, avg pool)
- **Goal fusion:** Early concatenation (obs + goal images along channel dim -> 6-channel input)
- **Policy head:** MLP (256, 256, 256) -> Gaussian with fixed std=1
- **Loss:** Negative log-likelihood (= MSE + constant when std is fixed)
- **Observations:** Head camera RGB only, resized to 256x256
- **Proprioception:** 23-dim (default) or 37-dim (with EEF), optional bounds normalization
- **Parameters:** ~21.6M (without proprio), ~21.8M (with 23-dim proprio)

See `goal-conditioned BC.md` for a detailed architecture description.

### GPU memory

| Batch size | Image size | Approx. GPU memory |
|---|---|---|
| 32 | 256x256 | ~4 GB |
| 128 | 256x256 | ~10 GB |
| 256 | 256x256 | ~20 GB |
| 1024 | 256x256 | ~80 GB |
| 2048 | 256x256 | ~160 GB |
| 3072 | 256x256 | ~240 GB |

### Large-GPU example (e.g., A100 80GB, H100 80GB, or multi-GPU with 275GB)

With a larger batch size, scale LR proportionally (linear scaling rule) and reduce total steps to keep the same data budget:

```bash
# BS=2048 on ~275GB GPU (e.g., 4xA100 or similar)
# LR scaled 8x: 3e-4 * (2048/256) = 2.4e-3
# Same data budget as BS=256/50K steps: 50000 * 256 / 2048 = 6250 steps
python -m gcbc_jax.train \
    --tfrecord_dir gcbc_jax/tfrecords/task-0053-final \
    --save_dir outputs/gcbc_jax_task0053_proprio_largegpu \
    --num_steps 6250 \
    --batch_size 2048 \
    --lr 2.4e-3 \
    --warmup_steps 250 \
    --use_proprio --normalize_proprio \
    --eval_interval 625 --save_interval 625 --log_interval 10 \
    --use_wandb --wandb_project gcbc-ispatialgym --run_name gcbc_proprio_bs2048
```

Or equivalently in epoch mode:

```bash
python -m gcbc_jax.train \
    --tfrecord_dir gcbc_jax/tfrecords/task-0053-final \
    --save_dir outputs/gcbc_jax_task0053_proprio_largegpu \
    --num_epochs 452 \
    --batch_size 2048 \
    --lr 2.4e-3 \
    --warmup_steps 250 \
    --use_proprio --normalize_proprio \
    --eval_interval 50 --save_interval 50 --log_interval 5 \
    --use_wandb --wandb_project gcbc-ispatialgym --run_name gcbc_proprio_bs2048
```

### Steps per epoch reference

| Task | Episodes | Train transitions | Steps/epoch (BS=256) |
|---|---|---|---|
| task-0051 (camera_relocalization) | 300 | ~189K | ~739 |
| task-0053 (object_scaling) | 300 | ~28K | ~111 |

## 4. Evaluate

The eval wrapper (`gcbc_jax/eval_policy.py`) loads a trained JAX checkpoint and provides the `forward(obs) -> torch.Tensor(23,)` interface expected by `eval_ispatialgym.py`.

### Extract goal image from TFRecord

The eval script needs a goal image PNG. To use the same goal image as training (from the TFRecord), extract it first:

```python
import tensorflow as tf
from PIL import Image

tfrecord_path = "gcbc_jax/tfrecords/task-0053-final/val/episode_0053000000097000.tfrecord"
for raw_record in tf.data.TFRecordDataset(tfrecord_path):
    parsed = tf.io.parse_single_example(raw_record,
        {"goal_image": tf.io.FixedLenFeature([], tf.string)})
    goal = tf.io.parse_tensor(parsed["goal_image"], tf.uint8).numpy()
    Image.fromarray(goal).save("outputs/gcbc_jax_task0053_proprio/goal_image_097000.png")
    break
```

### Run evaluation

From the `OmniGibson` directory, using the `behavior` conda env (with `--train` deps installed):

```bash
cd ../../OmniGibson

python -m omnigibson.learning.eval_ispatialgym \
    --task-name object_scaling \
    --episode-id 0053000000097000 \
    --policy-checkpoint $PROJECT_ROOT/il/bridge_data_v2/outputs/gcbc_jax_task0053_proprio \
    --goal-image-path $PROJECT_ROOT/il/bridge_data_v2/outputs/gcbc_jax_task0053_proprio/goal_image_097000.png \
    --use-proprio --normalize-proprio \
    --headless --write-video \
    --max-steps 500
```

**Eval arguments:**

| Argument | Description |
|---|---|
| `--policy-checkpoint` | Directory containing `checkpoint_*` files and `action_proprio_metadata.json` |
| `--goal-image-path` | Path to goal image PNG (defaults to episode's `reference_image2.png`) |
| `--use-proprio` | Must match training configuration |
| `--add-eef-proprio` | Must match training configuration |
| `--normalize-proprio` | Must match training configuration |
| `--headless` / `--no-headless` | Run with/without GUI |
| `--write-video` | Save evaluation video |

The eval policy automatically:
- Loads the latest `checkpoint_*` file from the checkpoint directory
- Extracts head camera RGB from Isaac Sim observations (drops alpha channel if RGBA)
- Extracts and normalizes proprioception from `robot_r1::proprio` (256-dim)
- Resizes observation images to match goal image resolution
- Denormalizes predicted actions using `action_proprio_metadata.json`
