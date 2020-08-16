import cv2
import numpy as np

import keras.backend as K
import keras.layers as layers
from keras import regularizers
from keras.applications import vgg16, resnet50
from keras.applications.vgg16 import VGG16
from keras.initializers import TruncatedNormal
from keras.layers import Input, BatchNormalization, Activation, AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, load_model
from keras.utils.data_utils import get_file
from Parameters import DEFAULT_ANCHORS_PER_LOC
from custom_layers import RoiResizeConv, Scale
from model_trainer import cls_loss_rpn, bbreg_loss_rpn

POOLING_REGIONS = 7
FINAL_CONV_FILTERS = 512 # 512 for VGG, 1024 for ResNet50
WEIGHT_REGULARIZER = None # None for VGG else regularizers.l2(5e-4)
BIAS_REGULARIZER = None # BIAS_REGULARIZER = None #
VGG_Stride = 16
ResNet_Stride = 16


"""
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
VGG 16
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
def vgg16_rpn_from_h5(h5_path, anchors_per_loc=DEFAULT_ANCHORS_PER_LOC):
    model_rpn = load_model(h5_path,
                           custom_objects={'cls_loss_rpn': cls_loss_rpn(anchors_per_loc=anchors_per_loc),
                                           'bbreg_loss_rpn': bbreg_loss_rpn(anchors_per_loc=anchors_per_loc)})

    return model_rpn


def vgg_get_conv_rows_cols(height, width):

    STRIDE = 16
    return height // STRIDE, width // STRIDE


def vgg_preprocess(data):
    rgb_data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    batched_rgb_data = np.expand_dims(rgb_data, axis=0).astype('float64')

    # The preprocess_input function is meant to adequate your image to the format the model requires.
    # It takes in a batch of images and outputs the same.
    # Keras works with batches of images. So, the first dimension is used for the number of samples (or images) you have.
    new_data = vgg16.preprocess_input(batched_rgb_data)[0] # [0]: To make the data in-built model compatible

    return new_data


def vgg16_base(freeze_blocks= [], weight_regularizer=None, bias_regularizer=None):
    img_input = Input(shape=(None, None, 3))

    # Block 1
    train1 = 1 not in freeze_blocks
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=train1,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=train1,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    train2 = 2 not in freeze_blocks
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=train2,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=train2,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    train3 = 3 not in freeze_blocks
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=train3,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=train3,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=train3,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    train4 = 4 not in freeze_blocks
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=train4,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=train4,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=train4,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5 (Conv operations are not decreasing the feature size. With 4 max Pooling operations, image size reduces by 2^4)
    train5 = 5 not in freeze_blocks
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=train5,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=train5,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=train5,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)

    base_model = Model(img_input, x, name='vgg16')

    return base_model


def vgg16_rpn(base_model, include_conv=False, weight_regularizer=None, bias_regularizer=None,
              anchors_per_loc=DEFAULT_ANCHORS_PER_LOC):
    gaussian_initializer = TruncatedNormal(stddev=0.01)
    net = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer=gaussian_initializer,
                 kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                 name='rpn_conv1', )(base_model.output)

    x_class = Conv2D(anchors_per_loc, (1, 1), activation='sigmoid',
                     kernel_initializer=gaussian_initializer,
                     kernel_regularizer=weight_regularizer,
                     bias_regularizer=bias_regularizer,
                     name='rpn_out_cls')(net)
    x_regr = Conv2D(anchors_per_loc * 4, (1, 1), activation='linear',
                    kernel_initializer=gaussian_initializer,
                    kernel_regularizer=weight_regularizer,
                    bias_regularizer=bias_regularizer,
                    name='rpn_out_bbreg')(net)

    outputs = [x_class, x_regr]
    if include_conv:
        outputs.append(base_model.output)
    rpn_model = Model(inputs=base_model.inputs, outputs=outputs)

    return rpn_model

def vgg16_classifier(num_rois, num_classes, base_model = None, weight_regularizer=None, bias_regularizer=None):
    """
    :param num_rois:
    :param num_classes:
    :param base_model: Set it None if the detector does not consists of a  base network, else pass it
    :param weight_regularizer:
    :param bias_regularizer:
    :return:
    """
    roi_input = Input(shape=(None, 4), name='roi_input') # From RPN
    pooling_input = base_model.output if base_model else Input(shape=(None, None, FINAL_CONV_FILTERS))

    ## roi_pooling_out.shape =  (1, num_rois, pool_size, pool_size, channels=512)
    roi_pooling_out = RoiResizeConv(POOLING_REGIONS, num_rois)([pooling_input, roi_input])

    ## This wrapper applies a layer to every temporal slice of an input
    out = TimeDistributed(Flatten(name='flatten'))(roi_pooling_out)                                                 # out will have a shape (1, num_rois, channels * pool_size * pool_size)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1',
                                kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer))(out)     # out will have a shape (1, num_rois, 4096)
    # out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2',
                                kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer))(out)
    # out = TimeDistributed(Dropout(0.5))(out)

    gaussian_initializer_class = TruncatedNormal(stddev=0.01)
    gaussian_initializer_reg = TruncatedNormal(stddev=0.001)

    # There are two output layer
    # out_class: softmax acivation function for classify the class name of the object
    # out_regr: linear activation function for bboxes coordinates regression
    out_class = TimeDistributed(Dense(num_classes, activation='softmax', kernel_initializer=gaussian_initializer_class,
                                      kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer),
                                name='dense_class_{}'.format(num_classes))(out)
    # note: no regression target for bg class
    out_reg = TimeDistributed(Dense(4 * (num_classes - 1), activation='linear', kernel_initializer=gaussian_initializer_reg,
                                    kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer),
                              name='dense_reg_{}'.format(num_classes))(out)

    model_input = base_model.input if base_model else pooling_input
    cls_model = Model(inputs=[model_input, roi_input], outputs=[out_class, out_reg])

    return cls_model

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------
ResNet 50
------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def resnet50_rpn_from_h5(h5_path, anchors_per_loc=DEFAULT_ANCHORS_PER_LOC):
    """
    Loads a saved rpn model from an h5 file.
    :param h5_path: string, filesystem path of the saved Keras model for the rpn.
    :param anchors_per_loc: positive integer, the number of used in the rpn saved in the file.
    :return: Keras model.
    """
    model_rpn = load_model(h5_path,
                           custom_objects={'cls_loss_rpn': cls_loss_rpn(anchors_per_loc=anchors_per_loc),
                                           'bbreg_loss_rpn': bbreg_loss_rpn(anchors_per_loc=anchors_per_loc),
                                           'Scale': Scale})

    return model_rpn


def resnet50_preprocess(data):
    """
    Convert raw bgr image to the format needed for pre-trained Imagenet weights to apply.
    :param data: numpy array containing bgr values of an image.
    :return: numpy array with preprocessed values.
    """
    # expect image to be passed in as BGR
    rgb_data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    batched_rgb_data = np.expand_dims(rgb_data, axis = 0).astype('float64')
    new_data = resnet50.preprocess_input(batched_rgb_data)[0]

    return new_data


def resnet50_get_conv_rows_cols(height, width):
    """
    Calculates the dimensions of the last conv4 layer for a given image size.
    :param height: positive integer, the image height in pixels.
    :param width: positive integer, the image width in pixels.
    :return: height and width of the last conv4 layer as a list of integers.
    """
    dims = [height, width]
    for i in range(len(dims)):
        # (3, 3) zeropad
        dims[i] += 6
        for filter_size in [7, 3, 1, 1]:
            # all strides use valid padding, formula is (W - F + 2P) / S + 1
            dims[i] = (dims[i] - filter_size) // 2 + 1

    return dims


def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True, use_conv_bias=True,
                   weight_regularizer=None, bias_regularizer=None, bn_training=False, separate_scale=False):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Additional arguments
        trainable: boolean for whether to make this block's layers trainable.
        use_conv_bias: boolean for whether or not convolutional layers should have a bias.
        weight_regularizer: keras.regularizers.Regularizer object for weight regularization on all layers, None if no
        regularization.
        bias_regularizer: keras.regularizers.Regularizer object for bias regularization on all layers, None if no
        regularization.
        bn_training: boolean for whether or not BatchNormalization layers should be trained. Should always be false as
        the model doesn't train correctly with batch normalization.
        separate_scale: boolean for whether or not the BatchNormalization layers should be followed by a separate Scale
        layer.
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'
    eps = 1e-5

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a', trainable=trainable, use_bias=use_conv_bias,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a',
                           trainable=bn_training)(x, training=bn_training)
    if separate_scale:
        x = Scale(axis=bn_axis, name=scale_name_base + '2a', trainable=bn_training)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable,
               use_bias=use_conv_bias,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b',
                           trainable=bn_training)(x, training=bn_training)
    if separate_scale:
        x = Scale(axis=bn_axis, name=scale_name_base + '2b', trainable=bn_training)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', trainable=trainable, use_bias=use_conv_bias,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c',
                           trainable=bn_training)(x, training=bn_training)
    if separate_scale:
        x = Scale(axis=bn_axis, name=scale_name_base + '2c', trainable=bn_training)(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True, use_conv_bias=True,
               weight_regularizer=None, bias_regularizer=None, bn_training=False, separate_scale=False):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Additional arguments
        trainable: boolean for whether to make this block's layers trainable.
        use_conv_bias: boolean for whether or not convolutional layers should have a bias.
        weight_regularizer: keras.regularizers.Regularizer object for weight regularization on all layers, None if no
        regularization.
        bias_regularizer: keras.regularizers.Regularizer object for bias regularization on all layers, None if no
        regularization.
        bn_training: boolean for whether or not BatchNormalization layers should be trained. Should always be false as
        the model doesn't train correctly with batch normalization.
        separate_scale: boolean for whether or not the BatchNormalization layers should be followed by a separate Scale
        layer.
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'
    eps = 1e-5

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable, use_bias=use_conv_bias,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a', trainable=bn_training)(x, training=bn_training)
    if separate_scale:
        x = Scale(axis=bn_axis, name=scale_name_base + '2a', trainable=bn_training)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable,
               use_bias=use_conv_bias, kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b', trainable=bn_training)(x, training=bn_training)
    if separate_scale:
        x = Scale(axis=bn_axis, name=scale_name_base + '2b', trainable=bn_training)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', trainable=trainable, use_bias=use_conv_bias,
               kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c', trainable=bn_training)(x, training=bn_training)
    if separate_scale:
        x = Scale(axis=bn_axis, name=scale_name_base + '2c', trainable=bn_training)(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1', trainable=trainable, use_bias=use_conv_bias,
                      kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1',
                                  trainable=bn_training)(shortcut, training=bn_training)
    if separate_scale:
        shortcut = Scale(axis=bn_axis, name=scale_name_base + '1', trainable=bn_training)(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet50_base(freeze_blocks=[1,2,3], weight_regularizer=None, bias_regularizer=None):
    """
    Creates a model of the ResNet-50 base layers used for both the RPN and detector.
    :param freeze_blocks: list of block numbers to make untrainable, e.g. [1,2,3] to not train the first 3 blocks.
    :param weight_regularizer: keras.regularizers.Regularizer object for weight regularization on all layers, None if no
    regularization.
    :param bias_regularizer: keras.regularizers.Regularizer object for bias regularization on all layers, None if no
        regularization.
    :return: Keras model for the base network.
    """
    img_input = Input(shape=(None, None, 3))
    bn_axis = 3
    train1 = 1 not in freeze_blocks
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', trainable=train1,
        kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', trainable=False)(x, training=False)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    train2 = 2 not in freeze_blocks
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=train2,
                   weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable=train2,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable=train2,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)

    train3 = 3 not in freeze_blocks
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable=train3,
                   weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable=train3,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable=train3,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable=train3,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)

    train4 = 4 not in freeze_blocks
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable=train4,
                   weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable=train4,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable=train4,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable=train4,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable=train4,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable=train4,
                       weight_regularizer=weight_regularizer, bias_regularizer=bias_regularizer)

    base_model = Model(img_input, x, name='resnet50')

    return base_model


def resnet50_rpn(base_model, load_weights = False, weight_regularizer=None, bias_regularizer=None, include_conv=False,
                 anchors_per_loc=DEFAULT_ANCHORS_PER_LOC):
    """
    Creates an rpn model on top of a passed in base model.
    :param base_model: Keras model returned by resnet50_base, containing only the first 4 blocks.
    :param weight_regularizer: keras.regularizers.Regularizer object for weight regularization on all layers, None if no
    regularization.
    :param bias_regularizer: keras.regularizers.Regularizer object for bias regularization on all layers, None if no
        regularization.
    :param include_conv: boolean for whether the conv4 output should be included in the model output.
    :param anchors_per_loc: number of anchors at each convolution position.
    :return: Keras model with the rpn layers on top of the base layers. Weights are initialized to Imagenet weights.
    """
    net = Conv2D(512, (3, 3), padding='same', activation='relu',kernel_initializer='normal',
                 kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                 name='rpn_conv1')(base_model.output)

    gaussian_initializer = TruncatedNormal(stddev=0.01)
    x_class = Conv2D(anchors_per_loc, (1, 1), activation='sigmoid', kernel_initializer=gaussian_initializer,
                     kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                     name='rpn_out_cls')(net)
    x_regr = Conv2D(anchors_per_loc * 4, (1, 1), activation='linear', kernel_initializer=gaussian_initializer,
                    kernel_regularizer=weight_regularizer, bias_regularizer=bias_regularizer,
                    name='rpn_out_bbreg')(net)

    outputs = [x_class, x_regr]
    if include_conv:
        outputs.append(base_model.output)

    rpn_model = Model(inputs = base_model.inputs, outputs = outputs)

    if load_weights:
        # weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        #                         WEIGHTS_PATH_NO_TOP,
        #                         cache_subdir='models',
        #                         md5_hash='a268eb855778b3df3c7506639542a6af')
        weights_path = "models/rpn_weights_{}_step1.h5".format("resnet50")
        rpn_model.load_weights(weights_path, by_name=True)
    return rpn_model