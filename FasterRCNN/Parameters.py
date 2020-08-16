from keras.optimizers import SGD, Adam
import math
import numpy as np

# Types of anchor
k = 9

# Default Values
DEFAULT_ANCHOR_SCALES = np.array([16, 32, 64, 128, 256, 512])
DEFAULT_ANCHOR_RATIOS = np.array([[1, 1], [1, 2], [2, 1]])
_NAIVE_ANCHORS = np.array([[size * height, size * width] for size in DEFAULT_ANCHOR_SCALES for height, width in DEFAULT_ANCHOR_RATIOS])
_RATIOS = np.array([math.sqrt(size * height * size * width) / size for size in DEFAULT_ANCHOR_SCALES for height, width in DEFAULT_ANCHOR_RATIOS])
DEFAULT_ANCHORS = (_NAIVE_ANCHORS // _RATIOS[:, None]).astype(int)
DEFAULT_ANCHORS_PER_LOC = len(DEFAULT_ANCHORS)
RESIZE_MIN_SIZE = 600
RESIZE_MAX_SIZE = 1000

# Resize dimensions
resize_min, resize_max = [600, 1000]

# Anchor Scales and Aspect Ratios
## Note that if im_size is smaller (than (600,_) or (_,1000)), anchor_box_scales should be scaled
anchor_scales = [128, 256, 512]
anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]


# Training Phases
DEFAULT_NUM_ITERATIONS = 10
phases = [[60000, 1e-3], [20000, 1e-4]] # From Paper, for training RPN. Make Sure num_iterations is a multiple of epoch length

# Optimizer
DEFAULT_LEARN_RATE = 1e-3
DEFAULT_MOMENTUM = 0.9
optimizer = SGD(lr=DEFAULT_LEARN_RATE, momentum=DEFAULT_MOMENTUM)
# [Optional]
# optimizer = Adam(lr=DEFAULT_LEARN_RATE)


NUM_ROIS = 64 # number of rois to be processed at a time by the detection/classification network
BBREG_MULTIPLIERS = np.array([10, 10, 5, 5], dtype=np.float32)

# If the box classification value is less than this, we ignore this box (testing)
bbox_threshold = 0.7