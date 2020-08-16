from __future__ import division
from model.Parameters import image_scaling_factor, image_subtraction_factor
from model.defineModel import retinanet_bbox
import numpy as np
import sys
import keras
import tensorflow as tf
import cv2
import warnings
import matplotlib


"""
------------------------------------------------Image Utility Functions-------------------------------------------------
"""


# # Not Required for Table Detection
def preprocess_image(x, mode='tf'):
    """ Preprocess an image by subtracting the ImageNet mean.
    Args
        x: np.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.
    Returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already

    # covert always to float32 to keep compatibility with opencv
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68
    elif mode == "custom_tf":
        # Distance transformed image of the table is saved and loaded using opencv, so the image values lie in [0,255]
        x /= image_scaling_factor
        x -= image_subtraction_factor
    return x


# For base anchor BBoxes, compute its shifter version according to feature map shape and stride
def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.
    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """
    shift_x = (keras.backend.arange(0, shape[1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride
    shift_y = (keras.backend.arange(0, shape[0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride

    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = keras.backend.reshape(shift_x, [-1])
    shift_y = keras.backend.reshape(shift_y, [-1])

    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = keras.backend.transpose(shifts)
    number_of_anchors = keras.backend.shape(anchors)[0]

    k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + keras.backend.cast(keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors


# Apply reg results to boxes after removing normalisation
def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    """ Applies deltas (usually regression results) to boxes (usually anchors).
    Before applying the deltas to the boxes, the normalization that was previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator. They are unnormalized in this function and then applied to the boxes.
    Args
        boxes : np.array of shape (B, N, 4), where B is the batch size, N the number of boxes and 4 values for (x1, y1, x2, y2).
        deltas: np.array of same shape as boxes. These deltas (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
        mean  : The mean value used when computing deltas (defaults to [0, 0, 0, 0]).
        std   : The standard deviation used when computing deltas (defaults to [0.2, 0.2, 0.2, 0.2]).
    Returns
        A np.array of the same shape as boxes, but with deltas applied to each box.
        The mean and std are used during training to normalize the regression values (networks love normalization).
    """
    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    width  = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

    pred_boxes = keras.backend.stack([x1, y1, x2, y2], axis=2)

    return pred_boxes


# The resize_image() function resizes one image at a time, but it does not update its object with
def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    """ Compute an image scale such that the image size is constrained to min_side and max_side.
    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.
    Returns
        A resizing scale.
    """
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def resize_image(img, min_side=800, max_side=1333):
    """ Resize an image such that the size is constrained to min_side and max_side.
    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.
    Returns
        A resized image.
    """
    # compute scale to resize the image
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return img, scale


# Resizing using imb objects as references
def resize_images(imgs, min_side=800, max_side=1333):
    """
    Resizes Multiple images (Using Image Objects for ref) such that the shorter side is min_size pixels, or the longer side is max_size pixels, whichever
    results in a smaller image.
    :param imgs: list of shape.Image objects to resize.
    :param min_side: minimum length in pixels of the shorter side.
    :param max_side: maximum length in pixels of the longer side.
    :return: list of resized images and list of resize ratio corresponding to each image.
    """
    resized_imgs = []
    resized_ratios = []

    for img in imgs:
        resized_img, resized_ratio = img.resize_within_bounds(min_size=min_side, max_size=max_side)
        resized_imgs.append(resized_img)
        resized_ratios.append(resized_ratio)

    return resized_imgs, resized_ratios


# The github implementation uses cython
# [https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/utils/compute_overlap.pyx]
def compute_overlap(boxes1, boxes2):
    """
    Optimized way of finding all the "intersection over union" overlaps between each box in one set with each box in
    another set. Much faster than calling calc_iou for each individual box pair. This function is optimized for the case
    where boxes2 is smaller than boxes1.
    :param boxes1: (anchor) 2-d numpy array where each row contains the [x1, y1, x2, y2] coordinates of a box.
    :param boxes2: (GT) 2-d numpy array where each row contains the [x1, y1, x2, y2] coordinates of a box.
    :return: 2-d numpy array with dimensions (m x n) where m is the length of boxes1 and n is the length of boxes2. The
     returned array is formatted such that result[i][j] is the IOU between box i of boxes1 and box j of boxes2.
    """
    result = np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    areas1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    areas2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # assume boxes2 is much smaller than boxes1 so iterate boxes2
    for i, box in enumerate(boxes2):
        x1_intersection = np.maximum(boxes1[:, 0], box[0])
        y1_intersection = np.maximum(boxes1[:, 1], box[1])
        x2_intersection = np.minimum(boxes1[:, 2], box[2])
        y2_intersection = np.minimum(boxes1[:, 3], box[3])

        w_intersection = np.maximum(0, x2_intersection - x1_intersection)
        h_intersection = np.maximum(0, y2_intersection - y1_intersection)

        area_intersection = w_intersection * h_intersection

        area_union = areas1 + areas2[i] - area_intersection

        result[:, i] = (area_intersection / (area_union))

    return result


"""
------------------------------------------------Model Utility Functions-------------------------------------------------
"""

def convert_model(model, nms=True, class_specific_filter=True, anchor_params=None):
    """ Converts a training model to an inference model.
    Args
        model                 : A retinanet training model.
        nms                   : Boolean, whether to add NMS filtering to the converted model.
        class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
        anchor_params         : Anchor parameters object. If omitted, default values are used.
    Returns
        A keras.models.Model object.
    Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    return retinanet_bbox(model=model, nms=nms, class_specific_filter=class_specific_filter, anchor_params=anchor_params)


def assert_training_model(model):
    """ Assert that the model is a training model.
    """
    assert(all(output in model.output_names for output in ['regression', 'classification'])), \
        "Input is not a training model (no 'regression' and 'classification' outputs were found, outputs are: {}).".format(model.output_names)


def check_training_model(model):
    """ Check that model is a training model and exit otherwise.
    """
    try:
        assert_training_model(model)
    except AssertionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


def freeze(model):
    """ Set all layers in a model to non-trainable.
    The weights for these layers will not be updated during training.
    This function modifies the given model in-place,
    but it also returns the modified model to allow easy chaining with other functions.
    """
    for layer in model.layers:
        layer.trainable = False
    return model


"""
------------------------------------------------Visualization Utility Functions-------------------------------------------------
"""


def label_color(label):
    """ Return a color from a set of predefined colors. Contains 80 colors in total.
    Args
        label: The label to get the color for.
    Returns
        A list of three values representing a RGB color.
        If no color is defined for a certain label, the color green is returned and a warning is printed.
    """

    colors = [list((matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0]) * 255).astype(int)) for x in np.arange(0, 1, 1.0 / 80)]
    if label < len(colors):
        return colors[label]
    else:
        warnings.warn('Label {} has no color, returning default.'.format(label))
        return 0, 255, 0


def extract_box(image, box):
    """
    Extract the region contained inside the box
    :param image:
    :param box:
    :return:
    """
    b = np.array(box).astype(int)
    sub_image = image[b[1]:b[3], b[0]:b[2]]
    return sub_image

def draw_box(image, box, color, thickness=5):
    """ Draws a box on an image with a given color.
    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    color = (color[0], color[1], color[2]) # Some why rectangle function is requiring color as int not like [0 255 0]
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), 0, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption, font_size=5, font_thickness=5):
    """ Draws a caption above the box in an image.
    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 255), font_thickness)


def draw_boxes(image, boxes, color, thickness=2):
    """ Draws boxes on an image with a given color.
    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


def draw_detections(image, boxes, scores, labels, color=None, label_to_name=None, score_threshold=0.5):
    """ Draws detections in an image.
    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        color           : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        c = color if color is not None else label_color(labels[i])
        draw_box(image, boxes[i, :], color=c)

        # draw labels
        caption = (label_to_name(labels[i]) if label_to_name else labels[i]) + ': {0:.2f}'.format(scores[i])
        draw_caption(image, boxes[i, :], caption)


def draw_annotations(image, annotations, color=(0, 255, 0), label_to_name=None):
    """ Draws annotations in an image.
    # Arguments
        image         : The image to draw on.
        annotations   : A [N, 5] matrix (x1, y1, x2, y2, label) or dictionary containing bboxes (shaped [N, 4]) and labels (shaped [N]).
        color         : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name : (optional) Functor for mapping a label to a name.
    """
    if isinstance(annotations, np.ndarray):
        annotations = {'bboxes': annotations[:, :4], 'labels': annotations[:, 4]}

    assert('bboxes' in annotations)
    assert('labels' in annotations)
    assert(annotations['bboxes'].shape[0] == annotations['labels'].shape[0])

    for i in range(annotations['bboxes'].shape[0]):
        label   = annotations['labels'][i]
        c       = color if color is not None else label_color(label)
        caption = '{}'.format(label_to_name(label) if label_to_name else label)
        draw_caption(image, annotations['bboxes'][i], caption)
        draw_box(image, annotations['bboxes'][i], color=c)
