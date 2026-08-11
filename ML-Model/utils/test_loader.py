import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

from utils.dataset_loader import (
    load_train_dataset,
    load_validation_dataset,
    load_test_dataset,
)

train_ds = load_train_dataset("skin")
val_ds = load_validation_dataset("skin")
test_ds = load_test_dataset("skin")

print(train_ds.class_names)
print(val_ds.class_names)
print(test_ds.class_names)