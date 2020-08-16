import tensorflow as tf
from keras import backend as K
from keras.objectives import categorical_crossentropy
from Parameters import DEFAULT_NUM_ITERATIONS, DEFAULT_LEARN_RATE, DEFAULT_ANCHORS_PER_LOC, BBREG_MULTIPLIERS, DEFAULT_ANCHORS

## Both cls and reg terms are roughly equally weighted in this way.
N_CLS = 256 #(normalized by the mini-batch size i.e. the number of samples considered by RPN (SAMPLE_SIZE = 256))
N_REG = 2400 #(reg term was normalized by the number of anchor locations ~2400)
LAMBDA_REG = 10.0 # BBox reg Loss multiplier used in the paper
LAMBDA_REG_DET = 1

"""
TIP: If the loss accepts additional parameters such as anchors_per_loc, make a nested function such that internal function 
does the usual computation on y_pred and y_true. The parameters passed to the outside function(for ex: cls_loss_rpn) 
will be available to the inside function(cls_loss_rpn_internal).

Comment: Binary Crossentropy computes the prob of object belonging to two different classes. Categorical CrossEntropy (two values)
measures the prob of an object belonging to a category (one value)
"""

def cls_loss_rpn(anchors_per_loc=DEFAULT_ANCHORS_PER_LOC):
    """
    Creates a loss function for the object classifying output of the RPN module.
        y_true.dim = [batch_size, conv_w, conv_h, anchors_per_loc*2]
        ":anchors_per_loc" selects first say 9 depth layers where every cell indicates whether to consider the anchor BBox
        "anchors_per_loc:" selects last say 9 depth layers where every cell indicates whether an object is present in the corresponding BBox or not
    :param anchors_per_loc: how many anchors at each convolution position.
    :return: a function used as a Keras loss function.
    """
    def cls_loss_rpn_internal(y_true, y_pred):
        selected_losses = y_true[:, :, :, :anchors_per_loc]
        y_is_pos = y_true[:, :, :, anchors_per_loc:]
        loss = K.sum(selected_losses * K.binary_crossentropy(y_is_pos, y_pred)) / N_CLS

        return loss

    return cls_loss_rpn_internal


def bbreg_loss_rpn(anchors_per_loc=DEFAULT_ANCHORS_PER_LOC):
    """
    Creates a loss function for the bounding box regression output of the RPN. Uses the "smooth" loss function defined
    in the paper. Smooth loss behaves like L1 loss when |x| is high (i.e. = |x| - 0.5 for |x| > alpha) to provide steady gradients and like L2
    loss to oscillate less for lower values of |x| (i.e. (0.5)*|x|^2 for |x| < alpha)
    :param anchors_per_loc: how many anchors at each convolution position.
    :return: a function used as a Keras loss function.
    """
    def bbreg_loss_rpn_internal(y_true, y_pred):
        selected_losses = y_true[:, :, :, :4 * anchors_per_loc]
        diff = y_true[:, :, :, 4 * anchors_per_loc:] - y_pred    # First ":" is for batch_size, second and third are for height and width
        abs_diff = K.abs(diff)
        multipliers_small = K.cast(K.less_equal(abs_diff, 1.0), tf.float32) # 1.0 is the value of hyper-param in the smooth L1 loss. Finding locations where |x|<1
        multipliers_big = 1.0 - multipliers_small # Finding where |x| > 1
        loss = LAMBDA_REG * selected_losses * K.sum(multipliers_small * (0.5 * abs_diff * abs_diff) + multipliers_big * (abs_diff - 0.5)) / N_REG

        return loss

    return bbreg_loss_rpn_internal


def bbreg_loss_det(num_classes):
    """
    Creates a loss function for the "smooth" bounding box regression output of the Fast R-CNN module. See the paper
    for details.
    :param num_classes: positive integer, the number of object classes used, NOT including background.
    :return: a function used as a Keras loss function.
    """
    def class_loss_internal(y_true, y_pred):
        # diff for bb reg
        x = y_true[:, :, 4*num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        # only consider loss from the ground truth class
        # use smooth L1 loss function
        loss = K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(1e-4 + y_true[:, :, :4*num_classes])
        return LAMBDA_REG_DET * loss
    return class_loss_internal


def cls_loss_det(y_true, y_pred):
    """
    Loss function for the object classification output of the Fast R-CNN module.
    :param num_classes: positive integer, the number of object classes used NOT including background.
    :return: tensor for the category cross entry loss.
    """
    return K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))