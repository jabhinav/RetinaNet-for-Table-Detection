import random
import timeit
import numpy as np
from Shapes import Box
from keras import backend as K
from Parameters import DEFAULT_NUM_ITERATIONS, DEFAULT_LEARN_RATE, DEFAULT_ANCHORS_PER_LOC, BBREG_MULTIPLIERS, DEFAULT_ANCHORS
from util import calc_iou, cross_ious, get_reg_params, get_bbox_coords, transform_np_inplace
from ROIUtils import _get_anchor_coords, _sanitize_boxes_inplace, _rois_to_truth, _get_valid_box_idxs, nms
from Losses import cls_loss_rpn, bbreg_loss_rpn, bbreg_loss_det, cls_loss_det

N_CLS = 256
N_REG = 2400

LAMBDA_REG = 10.0
LAMBDA_REG_DET = 1

POS_OVERLAP = 0.7
NEG_OVERLAP = 0.3

SAMPLE_SIZE = 256
MAX_POS_SAMPLES = 128

def _idx_to_conv(idx, conv_width, anchors_per_loc):
    """
    Converts an anchor box index in a 1-d numpy array to its corresponding 3-d index representing its convolution
    position and anchor index.
    :param idx: non-negative integer, the position in a 1-d numpy array of anchors.
    :param conv_width: the number of possible horizontal positions the convolutional layer's filters can occupy, i.e.
    close to the width in pixels divided by the cumulative stride at that layer.
    :param anchors_per_loc: positive integer, the number of anchors at each convolutional filter position.
    :return: tuple of the row, column, and anchor index of the convolutional filter position for this index.
    """
    divisor = conv_width * anchors_per_loc
    y, remainder = idx // divisor, idx % divisor
    x, anchor_idx = remainder // anchors_per_loc, remainder % anchors_per_loc
    return y, x, anchor_idx


def _num_boxes_to_conv_np(num_boxes, conv_width, anchors_per_loc):
    # similar to _idx_to_conv but for multiple boxes at once, uses vectorized operations to optimize the performance
    idxs = np.arange(num_boxes)
    divisor = conv_width * anchors_per_loc
    y, remainder = idxs // divisor, idxs % divisor
    x, anchor_idx = remainder // anchors_per_loc, remainder % anchors_per_loc
    return y, x, anchor_idx


def _get_conv_center(conv_x, conv_y, stride):
    """
    Finds the center of this convolution position in the image's original coordinate space.
    :param conv_x: non-negative integer, x coordinate of the convolution position.
    :param conv_y: non-negative integer, y coordinate of the convolution position.
    :param stride: positive integer, the cumulative stride in pixels at this layer of the network.
    :return: tuple of positive integers, the x and y coordinates of the center of the convolution position.
    """
    # (stride*conv_x, stride*conv_y) will take you to the top-left position, to get the center add half of the stride
    x_center = stride * (conv_x + 0.5)
    y_center = stride * (conv_y + 0.5)

    return int(x_center), int(y_center)


def _get_conv_center_np(conv_x, conv_y, stride):
    # like _get_conv_center but optimized for multiple boxes.
    x_center = stride * (conv_x + 0.5)
    y_center = stride * (conv_y + 0.5)

    return x_center.astype('int32'), y_center.astype('int32')


def _get_all_anchor_coords(conv_rows, conv_cols, anchor_dims, stride):
    """
    Given the shape of a convolutional layer and the anchors to generate for each position, return all anchors.
    :param conv_rows: positive integer, height of this convolutional layer.
    :param conv_cols: positive integer, width of this convolutional layer.
    :param anchor_dims: list of lists of positive integers, one height and width pair for each anchor.
    :param stride: positive integer, cumulative stride of this anchor position in pixels.
    :return: 2-d numpy array with one row for each anchor box containing its [x1, y1, x2, y2] coordinates.
    """
    num_boxes = conv_rows * conv_cols * len(anchor_dims)

    # for each anchor BBox associated to its cell in the conv layer, generate its conv x, y coordinate and id (out of 9=types of anchor at each conv cell)
    y, x, anchor_idxs = _num_boxes_to_conv_np(num_boxes, conv_cols, len(anchor_dims))
    x_center, y_center = _get_conv_center_np(x, y, stride)
    anchor_coords = np.zeros((num_boxes, 4), dtype=np.float32)
    anchor_height = anchor_dims[anchor_idxs, 0]
    anchor_width = anchor_dims[anchor_idxs, 1]

    anchor_coords[:, 0] = x_center - anchor_width // 2
    anchor_coords[:, 1] = y_center - anchor_height // 2
    anchor_coords[:, 2] = anchor_coords[:, 0] + anchor_width
    anchor_coords[:, 3] = anchor_coords[:, 1] + anchor_height

    return anchor_coords


def _get_out_of_bounds_idxs(anchor_coords, img_width, img_height):
    # internal function for figuring out which anchors are out of bounds
    out_of_bounds_idxs = np.where(np.logical_or.reduce((
        anchor_coords[:,0] < 0,
        anchor_coords[:,1] < 0,
        anchor_coords[:,2] >= img_width,
        anchor_coords[:,3] >= img_height)))[0]

    return out_of_bounds_idxs


def _apply_sampling(is_pos, can_use):
    """
    Applies the sampling logic described in the Faster R-CNN paper to determine which anchors should be evaluated in the
    loss function.
    :param is_pos: 1-d numpy array of booleans for whether each anchor is a true positive for some object.
    :param can_use: 1-d numpy array of booleans for whether each anchor can be used at all in the loss function.
    :return: 1-d numpy array of booleans of which anchors were chosen to be used in the loss function.
    """
    # extract (_)[0] due to np.where returning a tuple
    pos_locs = np.where(np.logical_and(is_pos == 1, can_use == 1))[0]
    neg_locs = np.where(np.logical_and(is_pos == 0, can_use == 1))[0]

    num_pos = len(pos_locs)
    num_neg = len(neg_locs)

    # cap the number of positive samples per batch to no more than half the batch size
    if num_pos > MAX_POS_SAMPLES:
        locs_off = random.sample(range(num_pos), num_pos - MAX_POS_SAMPLES) # Randomly sample excessive positive IDs
        can_use[pos_locs[locs_off]] = 0
        num_pos = MAX_POS_SAMPLES

    # fill remaining portion of the batch size with negative samples
    if num_neg + num_pos > SAMPLE_SIZE:
        locs_off = random.sample(range(num_neg), num_neg + num_pos - SAMPLE_SIZE)
        can_use[neg_locs[locs_off]] = 0

    return can_use


def _get_det_samples(is_pos, num_desired_rois, rpn_accuracy_for_epoch):
    """
    Applies the sampling logic described in the Fast R-CNN paper for one mini-batch: sample 64 regions of interest total
    of which 25% should be positive.
    :return: Ids of selected positive and negative samples from the list 'is_pos'. Total Ids = num_desired_rois
    """

    desired_pos = num_desired_rois // 4 # /4 since 25% should be positive.
    pos_samples = np.where(is_pos)
    neg_samples = np.where(np.logical_not(is_pos))

    if len(neg_samples) > 0:
        neg_samples = neg_samples[0]
    else:
        neg_samples = []

    if len(pos_samples) > 0:
        pos_samples = pos_samples[0]
    else:
        pos_samples = []

    rpn_accuracy_for_epoch.append((len(pos_samples)))

    if len(pos_samples) == 0:
        selected_pos_samples = []
    elif len(pos_samples) < desired_pos:
        # num_copies = desired_pos // len(pos_samples) + 1
        # selected_pos_samples = np.tile(pos_samples, num_copies)[:desired_pos].tolist()
        selected_pos_samples = pos_samples.tolist()
    else:
        selected_pos_samples = np.random.choice(pos_samples, desired_pos, replace=False).tolist()

    desired_neg = num_desired_rois - len(selected_pos_samples)

    if len(neg_samples) == 0:
        selected_neg_samples = []
    elif len(neg_samples) < desired_neg:
        # num_copies = desired_neg // len(neg_samples) + 1
        # selected_neg_samples = np.tile(neg_samples, num_copies)[:desired_neg].tolist()
        selected_neg_samples = np.random.choice(neg_samples, desired_neg, replace=True).tolist()
    else:
        selected_neg_samples = np.random.choice(neg_samples, desired_neg, replace=False).tolist()

    if len(selected_neg_samples) == 0 and len(pos_samples) > 0:
        num_copies = desired_neg // len(pos_samples) + 1
        selected_neg_samples = np.tile(pos_samples, num_copies)[:desired_neg].tolist()

    selected_samples = selected_pos_samples + selected_neg_samples

    return selected_samples, rpn_accuracy_for_epoch


class RpnTrainingManager:
    """
    Encapsulates the details of generating training inputs for a region proposal network for a given image.
    """

    def __init__(self, calc_conv_dims, stride, preprocess_func, anchor_dims=DEFAULT_ANCHORS):
        """
        :param calc_conv_dims: function that accepts a tuple of the image's height and width in pixels and returns the
        height and width of the convolutional layer prior to the rpn layers.
        :param stride: positive integer, the cumulative stride at the convolutional layer prior to the rpn layers.
        :param preprocess_func: function that applies the same transformation to the image's pixels as used for Imagenet
        training. Otherwise the Imagenet pre-trained weights will be mismatched.
        :param anchor_dims: list of lists of positive integers, one height and width pair for each anchor.
        """
        self._cache = {}
        self.calc_conv_dims = calc_conv_dims
        self.stride = stride
        self.preprocess_func = preprocess_func
        self.anchor_dims = anchor_dims

    def batched_image(self, image):
        """
        Returns the image data to be fed into the network.
        :param image: shapes.Image object.
        :return: 4-d numpy array with a single batch of the image, should can be used as a Keras model input.
        """
        return np.expand_dims(self.preprocess_func(image.data), axis=0)

    def _process(self, image):

        """
        Generates for every image - usable anchor BBoxes, positive(object present) BBoxes and their BBox regression w.r.t GT object BBoxes (in image coordinate space)
        Step1: Convert a list of shape.GroundTruthBox objects to a numpy array.
        Step2: Retrieve BBox image co-ords for each anchor (in image coordinate space)
        Step3: collect the ids of the anchor BBoxes going out of bound (in image coordinate space)
        Step4: Compute IOUs for each anchor and GT BBox pair
        Step5: Retrieve anchor BBoxes with max overlap and ones with IOU >0.7 and combine (unique)
        :param image: shapes.Image object

        can_use: 0 or 1 (0 means the box is invalid, 1 means the box is valid)
        is_pos:  0 or 1 (0 means the box is not an object, 1 means the box is an object)
        bbreg_targets: For positives anchors only (loop iterates over 'total_pos_idxs'), obtain their BBox regressions with GT BBoxes
        """

        # internal method, performs the expensive calculations needed to produce training inputs.
        conv_rows, conv_cols = self.calc_conv_dims(image.height, image.width) # # calculate the output map size based on the network architecture
        num_anchors = conv_rows * conv_cols * len(self.anchor_dims)

        # initialise empty output objectives
        bbreg_targets = np.zeros((num_anchors, 4), dtype=np.float32)
        can_use = np.zeros(num_anchors, dtype=np.bool)
        is_pos = np.zeros(num_anchors, dtype=np.bool)

        ## Step 1
        gt_box_coords = get_bbox_coords(image.gt_boxes)
        ## Step 2
        anchor_coords = _get_all_anchor_coords(conv_rows, conv_cols, self.anchor_dims, self.stride)
        ## Step 3
        out_of_bounds_idxs = _get_out_of_bounds_idxs(anchor_coords, image.width, image.height)
        ## Step 4
        all_ious = cross_ious(anchor_coords, gt_box_coords) # (num_anchors, num_gt_boxes) with each entry indicating their IOUs
        # all_ious, out_of_bounds_idxs = get_all_ious_faster(gt_box_coords, conv_rows, conv_cols, ANCHORS_PER_LOC, image.width, image.height, self.stride)

        ## Step 5
        max_iou_by_anchor = np.amax(all_ious, axis=1) # (num_anchors,) = max IOU between a selected anchor and all GT BBoxes
        max_idx_by_anchor = np.argmax(all_ious, axis=1) # (num_anchors,) = id of GT BBox having max IOU with selected anchor
        max_iou_by_gt_box = np.amax(all_ious, axis=0) # (num_gt_boxes,) = max IOU between a selected GT BBox and all anchors
        max_idx_by_gt_box = np.argmax(all_ious, axis=0) # (num_gt_boxes,) = id of anchor having max IOU with selected GT BBox

        ## anchors with more than 0.7 IOU with a gt box are positives
        pos_box_idxs = np.where(max_iou_by_anchor > POS_OVERLAP)[0]         # Gives IDs of the max overlap anchors having IOUs>0.7 with GT BBoxes
        ## for each gt box, the highest non-zero IOU anchor is a positive
        eligible_idxs = np.where(max_iou_by_gt_box > 0.0)                   # Gives IDs of max overlap GT BBoxes having IOU>0 with anchor
        more_pos_box_idxs = max_idx_by_gt_box[eligible_idxs]                # For each GT BBox from the previous step, retrieve the anchor it has max overlap with

        total_pos_idxs = np.unique(np.concatenate((pos_box_idxs, more_pos_box_idxs)))
        can_use[total_pos_idxs] = 1
        is_pos[total_pos_idxs] = 1

        # don't bother optimizing, profiling showed this loop's runtime is negligible
        for box_idx in total_pos_idxs:
            y, x, anchor_idx = _idx_to_conv(box_idx, conv_cols, len(self.anchor_dims))              # For each anchor, retrieve its 3-D position i.e. its x,
                                                                                                    # y position in the conv layer and its anchor_type index

            x_center, y_center = _get_conv_center(x, y, self.stride)                                # Anchor center in the image coordinate space
            anchor_height, anchor_width = self.anchor_dims[anchor_idx]
            anchor_box = Box.from_center_dims_int(x_center, y_center, anchor_width, anchor_height)
            gt_box_idx = max_idx_by_anchor[box_idx]

            reg_params = get_reg_params(anchor_box.corners, gt_box_coords[gt_box_idx])
            bbreg_targets[box_idx, :] = BBREG_MULTIPLIERS * reg_params                              # The multiplication with BBREG_MULTIPLIERS scales the
                                                                                                    # reg_params s.t. they become (tx*10., ty*10., tw*5., th*5.)

        # For negative samples, they are usable(can_use == 1) but not positive (is_pos == 0), neutral samples both remain zeros
        neg_box_idxs = np.where(np.logical_and(is_pos == 0, max_iou_by_anchor < NEG_OVERLAP))[0]
        can_use[neg_box_idxs] = 1
        can_use[out_of_bounds_idxs] = 0

        self._cache[image.cache_key] = {
            'can_use': can_use,
            'is_pos': is_pos,
            'bbreg_targets': bbreg_targets
        }

    def rpn_y_true(self, image):
        """
        Takes an image and returns the Keras model inputs/GT for each image to train with.
        :param image: shapes.Image object to generate training inputs for.
        :return: (1) tuple where the first element is a numpy array of the ground truth network output for
        whether each anchor overlaps with an object (object/no object for each anchor BBox),
        and (2) the second element is a numpy array of the ground truth network output for the
        bounding box transformation parameters to transform each anchor into an object's bounding box.
        (3) Number of positive anchors used for training (out of 256)
        """
        if image.cache_key not in self._cache:
            self._process(image)

        results = self._cache[image.cache_key]
        # TODO: why is the cached result being deleted? Investigate whether restoring it improves training time.
        # del self._cache[image.cache_key]
        can_use = _apply_sampling(results['is_pos'], results['can_use'])
        conv_rows, conv_cols = self.calc_conv_dims(image.height, image.width)

        is_pos = np.reshape(results['is_pos'], (conv_rows, conv_cols, len(self.anchor_dims)))
        can_use = np.reshape(can_use, (conv_rows, conv_cols, len(self.anchor_dims)))
        selected_is_pos = np.logical_and(is_pos, can_use)

        # Calculate number of positives (Dict = {True:__, False:__})
        unique, counts = np.unique(selected_is_pos, return_counts=True)
        num_pos = dict(zip(unique, counts)) # The dictionary has number of ones at [1] and number of zeros at [0]

        # combine arrays with whether or not to use for the loss function
        y_class = np.concatenate([can_use, is_pos], axis=2)
        bbreg_can_use = np.repeat(selected_is_pos, 4, axis = 2) # If the anchor is positive, so is their (tx,ty,tw,th). That's why repeat the one 4 times.
        bbreg_targets = np.reshape(results['bbreg_targets'], (conv_rows, conv_cols, 4 * len(self.anchor_dims)))
        y_bbreg = np.concatenate([bbreg_can_use, bbreg_targets], axis = 2)

        y_class = np.expand_dims(y_class, axis=0)               # For batch_size = 1
        y_bbreg = np.expand_dims(y_bbreg, axis=0)

        return y_class, y_bbreg, num_pos

class DetTrainingManager:

    def __init__(self, rpn_model, class_mapping, preprocess_func, num_rois=64, stride=16, anchor_dims=DEFAULT_ANCHORS):
        self.rpn_model = rpn_model
        self.class_mapping = class_mapping
        self.preprocess_func = preprocess_func
        self.num_rois = num_rois
        self.stride = stride
        self.anchor_dims = anchor_dims
        self._cache = {}
        self.conv_only = True if len(rpn_model.output) == 3 else False

    def batched_image(self, image):
        """
        Returns the image data to be fed into the network.
        :param image: shapes.Image object.
        :return: 4-d numpy array with a single batch of the image, should can be used as a Keras model input.
        """
        return np.expand_dims(self.preprocess_func(image.data), axis=0)

    def _out_from_image(self, batched_img):
        # gets rpn output on a batched image
        return self.rpn_model.predict_on_batch(batched_img)

    def _rois_from_image(self, image):
        # for a given image, return the relevant rpn outputs
        batched_img = self.batched_image(image)
        conv_out = None
        if self.conv_only:
            cls_out, regr_out, conv_out = self._out_from_image(batched_img)
        else:
            cls_out, regr_out = self._out_from_image(batched_img)

        conv_rows, conv_cols = regr_out.shape[1:3]  # regr_out.shape[0] is 1 which is the batch size
        anchor_coords = _get_anchor_coords(conv_rows, conv_cols, self.anchor_dims // self.stride).reshape((-1, 4))  # Reshaping operation creates the array of size (conv_rows*conv_cols*9=num of anchors, 4)
        reg_targets = regr_out[0].reshape((-1, 4))
        rois = transform_np_inplace(anchor_coords, reg_targets / BBREG_MULTIPLIERS)
        roi_coords = _sanitize_boxes_inplace(conv_cols, conv_rows, rois)
        roi_probs = cls_out.reshape((-1))  # Predictions are now in a 1-D array
        return roi_coords, roi_probs, conv_out

    def _process(self, image):
        """
        :param image:
        :set: cache_obj- 'rois': eligibile rois, 'y_class_num': one-hot-vector encoding (num_eligible_rois, num_classes),
        'y_transform': concatenated(labels((len(rois), 4 * num_classes), regTargets((len(rois), 4 * num_classes))
        """
        # internal method, performs the expensive calculations needed to produce training inputs (ROI coords are unscaled w.r.t BBREG_MULTIPLIERS).
        roi_coords, roi_probs, conv_out = self._rois_from_image(image)

        # TODO: get rid of this? already sanitized during previous step
        valid_idxs = _get_valid_box_idxs(roi_coords)
        roi_coords, roi_probs = roi_coords[valid_idxs], roi_probs[valid_idxs]
        # TODO: filtering out improbable ROIs would speed up NMS significantly, check if it hurts training results
        sorted_idxs = roi_probs.argsort()[::-1]
        # decreasing the number of boxes improves nms compute time
        truncated_idxs = sorted_idxs[0:12000]
        roi_coords, roi_probs = roi_coords[truncated_idxs], roi_probs[truncated_idxs]
        # casting to short ints cuts nms compute time by ~25%
        roi_coords = roi_coords.astype('int16')
        nms_rois, _ = nms(roi_coords, roi_probs, max_boxes=2000, overlap_thresh=0.7)

        # For each ROI, see its eligibility (i.e. whether it has min overlap of 0.1), if eligible: compute its class label (one-hot vector encoding) and
        # BBox Reg w.r.t GT BBox (if positive else set it to 0 for 'bg' class)
        filtered_rois, y_class_num, y_transform = _rois_to_truth(nms_rois, image, self.class_mapping,
                                                                 stride=self.stride)

        cache_obj = {
            'rois': filtered_rois,
            'y_class_num': y_class_num,
            'y_transform': y_transform
        }
        if conv_out is not None:
            cache_obj['conv_out'] = conv_out
        self._cache[image.cache_key] = cache_obj

    def get_training_input(self, image, rpn_accuracy_for_epoch):
        """
        Takes an image and returns the Keras model inputs to train with.
        :param image: shapes.Image object for which to generate training inputs.
        :param rpn_accuracy_for_epoch: a list which contains num of positive samples generated by RPN at each iteration
        :return: tuple of 5 elements:
        1. The first input to the model. For step 2 of training it's the image's pixels after preprocessing. For step 4
        it's the convolutional features output by the last layer prior to the RPN module.
        2. 2-d numpy array containing the regions of interest selected for training. One row per region, formatted as
        [x1, y1, x2, y2] in coordinate space of the last convolutional layer prior to the RPN module.
        3. 2-d numpy array containing the one hot encoding of the object classes corresponding to each selected region.
        One row for each region containing a 1 in the column corresponding to the object class, 0 in all other columns.
        4. 3-d numpy array representing separate 2-d arrays:
          4a. one hot encoding of the object class for each selected region but with 4 copies of each number. This is
          used to determine which of the outputs in 4b should contribute to the loss function.
          4b. bounding box regression targets for each non-background object class, for each selected region. If there
          are 64 regions and 20 object classes, then this would be a 64 row numpy array with 80 columns, where columns
          0 through 3 inclusive contain the regression targets for object class 0, 4 through 7 inclusive contain the
          regression targets for object class 1, etc.
        5. rpn_accuracy_for_epoch : appended list with new entry for the "image"
        """
        if image.cache_key not in self._cache:
            self._process(image)

        results = self._cache[image.cache_key]

        if len(results['rois']) == 0:
            rpn_accuracy_for_epoch.append(0)
            return None, None, None, None, rpn_accuracy_for_epoch

        rois, y_class_num, y_transform = results['rois'], results['y_class_num'], results['y_transform']
        # in the one-hot encoding the last index is bg hence use it to check neg/pos, here 0 is not the object_class label but the absence of the object_class
        found_object = y_class_num[:, -1] == 0
        # Desired total number of samples = num_rois, which will contain atmost 25% positives
        sampled_idxs, rpn_accuracy_for_epoch = _get_det_samples(found_object, self.num_rois, rpn_accuracy_for_epoch)

        rois, y_class_num, y_transform = rois[sampled_idxs], y_class_num[sampled_idxs], y_transform[sampled_idxs]

        # feed in the image during step 2, feed in the saved conv features during step 4
        first_input = results['conv_out'] if self.conv_only else self.batched_image(image)

        # in step 4 of training, caching the conv features takes too much memory so don't cache anything
        if self.conv_only:
            del self._cache[image.cache_key]

        return first_input, \
               np.expand_dims(rois, axis=0), \
               np.expand_dims(y_class_num, axis=0), \
               np.expand_dims(y_transform, axis=0), \
               rpn_accuracy_for_epoch


def train_rpn(rpn_model, images, training_manager, optimizer, phases=[[DEFAULT_NUM_ITERATIONS, DEFAULT_LEARN_RATE]],
              save_frequency=None, save_weights_dest=None, save_model_dest=None):
    """
    Trains a region proposal network.
    :param rpn_model: Keras model for the rpn to be trained.
    :param images: sequence of shapes.Image objects used to train the network.
    :param training_manager: model_trainer.RpnTrainingManager to produce training inputs from images.
    :param optimizer: keras.optimizers.Optimizer implementation to be used. Doesn't need a preconfigured learning rate.
    :param phases: list of lists specifying the learning rate schedule, e.g. [[1000, 1e-3], [100, 1e-4]] 1000 iterations
    with learning rate 1e-3 followed by 100 iterations with learning rate 1e-4.
    :param save_frequency: positive integer specifying how many iterations occur between saving the model's state. Leave
    it as None to disable saving during training.
    :param save_weights_dest: the path to save model weights as an h5 file after each save_frequency iterations.
    :param save_model_dest: the path to save the Keras model as an h5 file after each save_frequency iterations.
    :return: the rpn passed in.
    """
    num_train = len(images)

    anchors_per_loc = len(training_manager.anchor_dims)
    for phase_num, phase in enumerate(phases):
        num_iterations, learn_rate = phase
        optimizer.lr = K.variable(learn_rate, name='lr')
        rpn_model.compile(optimizer=optimizer, loss=[cls_loss_rpn(anchors_per_loc=anchors_per_loc),
                                                     bbreg_loss_rpn(anchors_per_loc=anchors_per_loc)])

        print("Starting phase {} of training: {} iterations with learning rate {}".format(
            phase_num, num_iterations, learn_rate))

        for i in range(num_iterations):
            img_idx = (i + num_iterations * phase_num) % num_train
            if img_idx == 0:
                random.shuffle(images)

            img = images[img_idx]

            print('Starting phase {} iteration {} with learn rate {}, training on image {}, flipped status: {}'.format(
                phase_num, i, learn_rate, img.name, img.flipped))

            print('img size: {}x{}'.format(img.width, img.height))

            batched_img = training_manager.batched_image(img)
            y_class, y_bbreg, num_pos = training_manager.rpn_y_true(img)
            print('Shape of y_rpn_cls {}'.format(y_class.shape))
            print('Shape of y_rpn_regr {}'.format(y_bbreg.shape))
            print('Number of positive anchors for this image: %d' % num_pos)

            start_time = timeit.default_timer()
            loss_rpn = rpn_model.train_on_batch(batched_img, [y_class, y_bbreg])  # Runs a single gradient update on a single batch of data (Our batch Size=1 image and 256 anchors).
            print("model_rpn.train_on_batch time: ", timeit.default_timer() - start_time)
            print('loss_rpn: {}'.format(loss_rpn))

            if save_frequency and i % save_frequency == 0:
                if save_weights_dest is not None:
                    rpn_model.save_weights(save_weights_dest)
                    print('Saved rpn weights to {}'.format(save_weights_dest))
                if save_model_dest is not None:
                    rpn_model.save(save_model_dest)
                    print('Saved rpn model to {}'.format(save_model_dest))

    return rpn_model

def combined_rpn_det_trainer(rpn_model, detector, images, rpn_training_manager, det_training_manager, optimizer,
                         phases=[[DEFAULT_NUM_ITERATIONS, DEFAULT_LEARN_RATE]], save_frequency=None,
                         rpn_save_weights_dest=None, det_save_weights_dest=None, recordCSV = None, record_path=None):
    """
    Trains a Fast R-CNN object detector for step 2 of the 4-step alternate training scheme in the paper.
        :param detector: Keras model for the detector module. The model should accepts images and regions as inputs.
        :param rpn_model: Keras model for the rpn to be trained.
    :param images: sequence of shapes.Image objects used to train the network.
        :param rpn_training_manager: model_trainer.RpnTrainingManager to produce training inputs from images.
        :param det_training_manager: model_trainer.DetTrainingManager object produce training inputs from images.
    :param optimizer: keras.optimizers.Optimizer implementation to be used. Doesn't need a preconfigured learning rate.
    :param phases: list of lists specifying the learning rate schedule, e.g. [[1000, 1e-3], [100, 1e-4]] 1000 iterations
    with learning rate 1e-3 followed by 100 iterations with learning rate 1e-4.
    :param save_frequency: positive integer specifying how many iterations occur between saving the model's state
    (we are using its value as an epoch length). Leave it as None to disable saving during training.
    :param rpn_save_weights_dest: the path to save model weights as an h5 file after each save_frequency iterations.
    :param det_save_weights_dest: ---------- Leave them as None to disable weight saving during training ------------
    :param recordCSV:
    :param record_path:
    :return: trained rpn_model, detector passed in.
    """
    num_train = len(images)
    num_classes = len(det_training_manager.class_mapping) - 1
    anchors_per_loc = len(rpn_training_manager.anchor_dims)

    ## Secondary Parameters Initialization
    best_loss = np.Inf
    epoch_length = save_frequency if save_frequency else 2000
    # losses = np.zeros((epoch_length, 5))
    # rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    verbose = True

    for phase_num, phase in enumerate(phases):
        num_iterations, learn_rate = phase
        losses_rpn = np.zeros((3,)) # Reset them at the beginning of a new phase, only last phase's best loss will stay the same
        losses_det = np.zeros((4,))
        no_rois_samples = 0
        optimizer.lr = K.variable(learn_rate, name='lr')

        # Compilation of the model should be inside the loop since learning rate is changing at each phase.
        rpn_model.compile(optimizer=optimizer, loss=[cls_loss_rpn(anchors_per_loc=anchors_per_loc),
                                                     bbreg_loss_rpn(anchors_per_loc=anchors_per_loc)])
        detector.compile(optimizer=optimizer, loss=[cls_loss_det, bbreg_loss_det(num_classes)],
                         metrics={'dense_class_{}'.format(num_classes+1): 'accuracy'}) # For metrics, total classes including bg class

        print("----------------- RPN MODEL -----------------")
        print(rpn_model.summary())
        print("----------------- DET MODEL -----------------")
        print(detector.summary())

        print("\nStarting phase {} of training: {} iterations with learning rate {}".format(
            phase_num, num_iterations, learn_rate))

        for i in range(num_iterations):
            img_idx = (i + num_iterations * phase_num) % num_train
            # epoch_iterator_idx = i % epoch_length

            if img_idx == 0:
                random.shuffle(images)

            img = images[img_idx]

            print('\nStarting phase {} iteration {} with learn rate {}, training on image {}, flipped status: {}'.format(
                phase_num, i, learn_rate, img.name, img.flipped))

            ## RPN TRAINING
            batched_img = rpn_training_manager.batched_image(img)
            y_class, y_bbreg, num_pos = rpn_training_manager.rpn_y_true(img)
            rpn_start_time = timeit.default_timer()
            loss_rpn = rpn_model.train_on_batch(batched_img, [y_class, y_bbreg])  # For 1 image with 256 anchors, loss_rpn = [ total_loss, cls_loss_rpn, bbreg_loss_rpn]
            print("Number of positive/negative proposals: {}".format(num_pos))
            print("model_rpn.train_on_batch time: ", timeit.default_timer() - rpn_start_time)
            print('loss_rpn: {}'.format(loss_rpn))
            losses_rpn += loss_rpn

            ## DETECTOR TRAINING (FastRCNN Network) [NOTE: y_transform i.e. reg target for det Network is unscaled w.r.t BBREG_MULTIPLIERS]
            batched_img, rois, y_class_num, y_transform, rpn_accuracy_for_epoch = det_training_manager.get_training_input(img, rpn_accuracy_for_epoch)
            if rois is None:
                print("Found no rois for this image.")
                no_rois_samples += 1
            else:
                det_start_time = timeit.default_timer()
                loss_det = detector.train_on_batch([batched_img, rois], [y_class_num, y_transform]) # loss_det = [ total_loss, loss_class_cls, loss_class_regr, class_acc]
                print("model_det.train_on_batch time: ", timeit.default_timer() - det_start_time)
                print('loss_det: {}'.format(loss_det))
                losses_det += loss_det

            ## Not using below to record losses, since for samples with no rois, loss_det does not get updated
            # losses[epoch_iterator_idx, 0] = loss_rpn[1]
            # losses[epoch_iterator_idx, 1] = loss_rpn[2]
            # losses[epoch_iterator_idx, 2] = loss_det[1]
            # losses[epoch_iterator_idx, 3] = loss_det[2]
            # losses[epoch_iterator_idx, 4] = loss_det[3]

            ## SAVE MODEL AND PRINT STATEMENTS AT EPOCH END
            if save_frequency and i > 0 and (i+1) % epoch_length == 0:

                # loss_rpn_cls = np.mean(losses[:, 0])
                # loss_rpn_regr = np.mean(losses[:, 1])
                # loss_class_cls = np.mean(losses[:, 2])
                # loss_class_regr = np.mean(losses[:, 3])
                # class_acc = np.mean(losses[:, 4])

                loss_rpn_cls = losses_rpn[1]/epoch_length
                loss_rpn_regr = losses_rpn[2]/epoch_length
                loss_class_cls = losses_det[1]/(epoch_length-no_rois_samples)
                loss_class_regr = losses_det[2]/(epoch_length-no_rois_samples)
                class_acc = losses_det[3]/(epoch_length-no_rois_samples)

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)

                ## Reinitialise after epoch end
                rpn_accuracy_for_epoch = []
                losses_rpn = np.zeros((3,))
                losses_det = np.zeros((4,))
                no_rois_samples = 0
                # losses = np.zeros((epoch_length, 5))


                if verbose:
                    print('\nMean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                        mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Total loss: {}'.format(loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr))

                ## If loss at epoch end is lower than best min loss, save the weights
                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                if curr_loss < best_loss:
                    print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                    best_loss = curr_loss
                    if det_save_weights_dest is not None and rpn_save_weights_dest is not None:
                        detector.save_weights(det_save_weights_dest)
                        rpn_model.save_weights(rpn_save_weights_dest)
                        print('Saved detector weights to {}'.format(det_save_weights_dest))

                ## Save the losses and metrics at epoch end
                new_row = {'mean_overlapping_bboxes': round(mean_overlapping_bboxes, 3),
                           'class_acc': round(class_acc, 3),
                           'loss_rpn_cls': round(loss_rpn_cls, 3),
                           'loss_rpn_regr': round(loss_rpn_regr, 3),
                           'loss_class_cls': round(loss_class_cls, 3),
                           'loss_class_regr': round(loss_class_regr, 3),
                           'curr_loss': round(curr_loss, 3),
                           'mAP': 0}
                recordCSV = recordCSV.append(new_row, ignore_index=True)
                recordCSV.to_csv(record_path, index=0)

    return rpn_model, detector