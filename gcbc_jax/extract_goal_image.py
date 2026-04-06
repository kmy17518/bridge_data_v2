"""Extract the goal_image from a single TFRecord and save as PNG."""

import sys
import tensorflow as tf
from PIL import Image
import numpy as np

TFRECORD_PATH = (
    "/home/minyeongk/Desktop/projects/behavior-1k-private/il/bridge_data_v2/"
    "gcbc_jax/tfrecords/task-0051-final/val/episode_0051000000513000.tfrecord"
)
OUTPUT_PATH = (
    "/home/minyeongk/Desktop/projects/behavior-1k-private/il/bridge_data_v2/"
    "gcbc_jax/tfrecords/task-0051-final/val/episode_0051000000513000_goal.png"
)

# Read the single record from the TFRecord file
dataset = tf.data.TFRecordDataset(TFRECORD_PATH)
raw_record = next(iter(dataset))

# Parse the goal_image feature (stored as a serialized tensor of uint8)
features = tf.io.parse_single_example(
    raw_record,
    {"goal_image": tf.io.FixedLenFeature([], tf.string)},
)
goal_image = tf.io.parse_tensor(features["goal_image"], out_type=tf.uint8)

print(f"goal_image shape: {goal_image.shape}")
print(f"goal_image dtype: {goal_image.dtype}")

# Save as PNG
img = Image.fromarray(goal_image.numpy())
img.save(OUTPUT_PATH)
print(f"Saved to: {OUTPUT_PATH}")
