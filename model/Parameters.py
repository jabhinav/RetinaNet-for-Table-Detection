import numpy as np
import keras
import os
import tensorflow as tf

backbone_name = "resnet50"

# Training Paths (System set)
# root_dir = os.getcwd()
# ImageDir = os.path.join(os.path.join(root_dir, 'data'), 'trainV2')
# csv_path = os.path.join(os.path.join(root_dir, 'data'), 'trainV2.csv')
# tensorboard_dir = os.path.join(root_dir, 'tensorboard')
# snapshot_path = os.path.join(root_dir, "models")

# Paths for testing
root_dir = os.getcwd()
trained_model_path = os.path.join(root_dir, 'models', 'training-U7U2ycFZg_resnet50_48.h5')


# Pre-processing
image_min_side = 800
image_max_size = 1333
image_scaling_factor = 127.5        # Choose these factors based on the type of image (RGB or dis. transfom etc.)
image_subtraction_factor = 1.

# Training Parameters
# In paper, model is trained for 90k iterations with lr_initial=0.01 which is divided by 10 at 60k and again at 80k.
# SGD with weight decay of 0.0001 and momentum = 0.9
steps_per_epoch = 2000  # implementation used 10000
num_epochs = 100  # 50
learning_rate = 1e-4
batch_size = 1
multi_gpu = len(tf.config.list_physical_devices('GPU'))
multiprocessing = False             # Use multiprocessing in fit_generator.
num_workers = 1                     # Number of generator workers.
max_queue_size = 10                 # Queue length for multiprocessing workers in fit_generator (default=10)

# Anchor Parameters [currently set as default. To change, change here and also set config=True
# while creating object for train generator]
sizes = [32, 64, 128, 256, 512]   # 32 is the approx anchor size(w or h) for Pyramid Level 3 in the image space
strides = [8, 16, 32, 64, 128]      # At pyramid level 3 say, stride becomes 2**3,
ratios = np.array([0.5, 1, 2], keras.backend.floatx())
scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx())

# Miscellaneous Parameters
negative_overlap = 0.4
positive_overlap = 0.5
std = [0.2, 0.2, 0.2, 0.2]             # BBox reg divider to scale dx1, dy1, dx2, dy2
class_mapping = {"table": 0}        # Do not include a background class as it is implicit.


"""
How to compute BBoxes for different anchor types:
Given: Pyramid Level=3, size = 32
AnchorType 1: ratio = r, scale = f
            H_old, W_old = 32f
            H_new * W_new = (32f)^2
            H_new/W_new = r
            W_new = sqrt(Area/r), H_new = W_new*r

"""