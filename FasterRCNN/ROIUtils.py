import numpy as np
from util import transform_np_inplace, cross_ious, get_reg_params, get_bbox_coords
from Parameters import BBREG_MULTIPLIERS

CLASSIFIER_MIN_OVERLAP = 0.1
CLASSIFIER_POS_OVERLAP = 0.5

PROBABLE_THRESHOLD = 0.05

def _one_hot_encode_bbreg(rois, gt_boxes, is_pos, class_mapping):
    # finds the one hot encoded input for the bounding box regression part of the Fast-RCNN module prior to sampling. For positive ROIs, set the labels
    # to (1,1,1,1) and RegTargets to (tx, ty, tw, th) else set the labels and RegTargets to (0,0,0,0)
    num_classes = len(class_mapping) - 1

    targs = np.zeros((len(rois), 4 * num_classes), dtype=np.float32)
    labels = np.zeros((len(rois), 4 * num_classes), dtype=np.float32)

    for i, (roi, gt_box, pos) in enumerate(zip(rois, gt_boxes, is_pos)):
        if pos:
            class_idx = class_mapping[gt_box.obj_cls]
            labels[i, 4*class_idx:4*(class_idx+1)] = 1, 1, 1, 1

            tx, ty, tw, th = get_reg_params(roi, gt_box.corners) # Computed in feature coordinate space
            targs[i, 4*class_idx:4*(class_idx+1)] = tx, ty, tw, th
            targs[i, 4*class_idx:4*(class_idx+1)] *= BBREG_MULTIPLIERS

    return np.concatenate([labels, targs], axis=1)


def _one_hot_encode_cls(obj_classes, class_to_num):
    # finds the one hot encoded input for the region classification part of the Fast-RCNN module prior to sampling.
    num_classes = len(class_to_num)
    result = np.zeros((len(obj_classes), num_classes), dtype=np.int32)

    for i, obj_cls in enumerate(obj_classes):
        result[i, class_to_num[obj_cls]] = 1

    return result


def _get_valid_box_idxs(boxes):
    # find the boxes with positive width and height.
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    valid_idxs = np.where((x2 > x1) & (y2 > y1))[0]

    return valid_idxs


def nms(boxes, probs, overlap_thresh=0.7, max_boxes=300):
    """
    Applies non-maximum suppression to a set of boxes and their probabilities of containing an object.
    :param boxes: 2-d numpy array, one row for each box containing its [x1, y1, x2, y2] coordinates.
    :param probs: 1-d numpy array of floating point numbers, probability that the box with this index is an object.
    :param overlap_thresh: floating point number, a fraction indicating the minimum overlap between two boxes needed to
    suppress the one with lower probability.
    :param max_boxes: positive integer, how many output boxes desired.
    :return: tuple of 2 numpy arrays, the selected boxes and their probabilities in the same format as the inputs.
    """

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by their object containing probabilities
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:

        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box (Here we do this for
        # all indexes in the indexes list at once without looping over)
        x1_intersection = np.maximum(x1[i], x1[idxs[:last]])
        y1_intersection = np.maximum(y1[i], y1[idxs[:last]])
        x2_intersection = np.minimum(x2[i], x2[idxs[:last]])
        y2_intersection = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w_intersection = np.maximum(0, x2_intersection - x1_intersection + 1)
        h_intersection = np.maximum(0, y2_intersection - y1_intersection + 1)

        area_intersection = w_intersection * h_intersection

        area_union = area[i] + area[idxs[:last]] - area_intersection
        overlap = area_intersection / area_union

        # if there is sufficient overlap, suppress the
        # corresponding bounding boxes and retain the
        # remaining ones
        idxs = idxs[np.where(overlap <= overlap_thresh)[0]]

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked
    return boxes[pick], probs[pick]


def _rois_to_truth(rois, image, class_mapping, stride=16):
    # for an image, some regions of interest, and object classes, find the ground truth prior to sampling
    gt_boxes = [gt_box.resize(1/stride) for gt_box in image.gt_boxes]# Rescale GTBBox coords so that it lies in feature coordinate space
    gt_coords = get_bbox_coords(gt_boxes)                            # 2-d numpy array where each row contains the [x1, y1, x2, y2] coordinates of a box.
    all_ious = cross_ious(rois, gt_coords)
    max_iou_by_roi = np.amax(all_ious, axis=1)
    max_gt_by_roi = np.argmax(all_ious, axis=1)
    eligible_roi_idxs = np.where(max_iou_by_roi >= CLASSIFIER_MIN_OVERLAP)[0]
    eligible_rois = rois[eligible_roi_idxs]

    pos_idxs = np.where(max_iou_by_roi >= CLASSIFIER_POS_OVERLAP)[0]

    ## For eligible and pos ROI, retrieve its corresponding GT Box, else for remaining eligible ROIs, set the GT Box to None
    eligible_gt_boxes = [gt_boxes[max_gt_by_roi[i]] if i in pos_idxs else None for i in eligible_roi_idxs]
    obj_classes = [box.obj_cls if box else 'bg' for box in eligible_gt_boxes] # For None GT BBoxes, set its class as 'bg'

    # Encoded Labels array of size (num_objects, num_object_classes) with 1 at the object class in each row(using class_mapping)
    encoded_labels = _one_hot_encode_cls(obj_classes, class_mapping)

    bbreg_labels_targets = _one_hot_encode_bbreg(
        eligible_rois,
        eligible_gt_boxes,
        is_pos=np.isin(eligible_roi_idxs, pos_idxs),
        class_mapping=class_mapping
    )

    return eligible_rois, encoded_labels, bbreg_labels_targets


def _sanitize_boxes_inplace(conv_cols, conv_rows, coords):
    # clip portions of regions lying outside the feature map boundaries.

    # set minimum width/height to 1
    coords[:, 2] = np.maximum(coords[:, 0] + 1, coords[:, 2])
    coords[:, 3] = np.maximum(coords[:, 1] + 1, coords[:, 3])
    # x1 and y1 must be at least 0
    coords[:, 0] = np.maximum(0, coords[:, 0])
    coords[:, 1] = np.maximum(0, coords[:, 1])
    # x2 and y2 must be at most cols-1 and rows-1
    coords[:, 2] = np.minimum(conv_cols - 1, coords[:, 2])
    coords[:, 3] = np.minimum(conv_rows - 1, coords[:, 3])

    return coords


def _get_anchor_coords(conv_rows, conv_cols, anchor_dims, multiplier=1):
    """
    Returns the [x1, y1, x2, y2] coordinates of anchors for each convolution position(coordinates correspond to the last conv layer not the image, that's why anchor_dims // stride).
    :param conv_rows: Num of rows in the last conv layer
    :param conv_cols: Num of columns in the last conv layer
    :param anchor_dims: Anchor dimensions resized to fit the last conv layer
    :param multiplier:
    :return anchor coords in the form x1, y1, x2, y2 in an array of shape (conv_rows, conv_cols, num_anchors, 4)
    """

    # Example, Create 18x25 mesh grid
    # For every point in x, there are all the y points and vice versa
    # x_center.shape = (18, 25)
    # y_center.shape = (18, 25)

    coords = np.zeros((conv_rows, conv_cols, len(anchor_dims), 4), dtype=np.float32)
    x_center, y_center = np.meshgrid(np.arange(conv_cols), np.arange(conv_rows))

    for i, anchor in enumerate(anchor_dims):
        anchor_height, anchor_width = anchor * multiplier

        coords[:, :, i, 0] = x_center - anchor_width // 2
        coords[:, :, i, 1] = y_center - anchor_height // 2
        coords[:, :, i, 2] = coords[:, :, i, 0] + anchor_width
        coords[:, :, i, 3] = coords[:, :, i, 1] + anchor_height

    return coords


def get_rois_from_image(rpn_model, batched_img, anchor_dims, stride, include_conv=False):
    """
    :param rpn_model: The RPN model with weights loaded
    :param batched_img: Image (not img object) with added dimension for batch size =1
    :param anchor_dims: Your dimensions for anchor BBoxes
    :param stride: Network Stride
    :param include_conv: Whether last feature map of the shared layer was included in the output
    :return: tuple with first element an array (shape = (total_predictions, 4)) of the predicted BBox coordinates in the form of
    x1, y1, x2, y2. Second element is the prediction probabilities for each BBox. Third element is the last feature map if included
    in the output of the network
    """
    conv_out = None
    if include_conv:
        cls_out, regr_out, conv_out = rpn_model.predict_on_batch(batched_img)
    else:
        cls_out, regr_out = rpn_model.predict_on_batch(batched_img)  # cls_out: shape (1, feature_map.height, feature_map.width, num_anchors)
                                                                     # regr_out: shape (1, feature_map.height, feature_map.width, num_anchors*4)

    # turn the rpn's regression output and anchor dimensions into regions.
    ## DOUBT: Why divide by stride. Predictions are in image coordinate space, so should the anchor_coords.

    conv_rows, conv_cols = regr_out.shape[1:3]  # regr_out.shape[0] is 1 which is the batch size
    anchor_coords = _get_anchor_coords(conv_rows, conv_cols, anchor_dims // stride).reshape((-1, 4))  # Reshaping operation creates the array of size (conv_rows*conv_cols*9=num of anchors, 4)
    reg_targets = regr_out[0].reshape((-1, 4))
    rois = transform_np_inplace(anchor_coords, reg_targets / BBREG_MULTIPLIERS)
    roi_coords = _sanitize_boxes_inplace(conv_cols, conv_rows, rois)
    roi_probs = cls_out.reshape((-1))  # Predictions are now in a 1-D array

    return roi_coords, roi_probs, conv_out


def RPNsToROIs(rpn_model, batched_img, anchor_dims, stride, include_conv= False):

    """
    Convert rpn layer to roi bboxes
    :param rpn_model: The RPN model with weights loaded
    :param batched_img: Image (not img object) with added dimension for batch size =1
    :param anchor_dims: Your dimensions for anchor BBoxes
    :param stride: Network Stride
    :param include_conv: Whether last feature map of the shared layer was included in the output
    :return: tuple containing boxes from non-max-suppression (shape=(max_boxes, 4)) and their prediction probabilities
    """

    roi_coords, roi_probs, conv_out = get_rois_from_image(rpn_model, batched_img, anchor_dims, stride, include_conv)

    # Find out the bboxes which are illegal and delete them from bboxes list
    valid_idxs = _get_valid_box_idxs(roi_coords)
    roi_coords, roi_probs = roi_coords[valid_idxs], roi_probs[valid_idxs]

    ## TODO: filtering out improbable ROIs would speed up NMS significantly, check if it hurts training results
    sorted_idxs = roi_probs.argsort()[::-1]                                                         # Outputs the sorted indexes in dec order

    ## decreasing the number of boxes improves nms compute time
    truncated_idxs = sorted_idxs[0:12000]
    roi_coords, roi_probs = roi_coords[truncated_idxs], roi_probs[truncated_idxs]

    ## casting to short ints cuts nms compute time by ~25%
    roi_coords = roi_coords.astype('int16')
    nms_rois, nms_rois_probs = nms(roi_coords, roi_probs, max_boxes=300, overlap_thresh=0.7)

    ## Use it when RPN network is appended with classification network of Faster-RCNN - Generates GT regions with object type annotations for classification network
    # filtered_rois, y_class_num, y_transform = _rois_to_truth(nms_rois, image, self.class_mapping, stride=self.stride)

    # cache_obj = {
    #     'rois': filtered_rois,
    #     'y_class_num': y_class_num,
    #     'y_transform': y_transform
    # }
    # if conv_out is not None:
    #     cache_obj['conv_out'] = conv_out
    # self._cache[image.cache_key] = cache_obj
    return nms_rois, nms_rois_probs