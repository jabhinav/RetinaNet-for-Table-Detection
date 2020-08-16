import keras
import keras_resnet
import keras_resnet.models
from model import losses, utils, initializers, layers
from model.anchors import AnchorParameters_default


# Backbone Class over
class Backbone:
    """ This class stores additional information on backbones.
        """

    def __init__(self, backbone):
        # a dictionary mapping custom layer names to the correct classes
        self.custom_objects = {
            'UpsampleLike': layers.UpsampleLike,
            'PriorProbability': initializers.PriorProbability,
            'RegressBoxes': layers.RegressBoxes,
            'FilterDetections': layers.FilterDetections,
            'Anchors': layers.Anchors,
            'ClipBoxes': layers.ClipBoxes,
            '_smooth_l1': losses.smooth_l1(),
            '_focal': losses.focal(),
        }

        self.backbone = backbone
        self.validate()

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        raise NotImplementedError('retinanet method not implemented.')

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        raise NotImplementedError('download_imagenet method not implemented.')

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        raise NotImplementedError('validate method not implemented.')

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        Having this function in Backbone allows other backbones to define a specific preprocessing step.
        """
        raise NotImplementedError('preprocess_image method not implemented.')

class ResNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(ResNetBackbone, self).__init__(backbone)
        self.custom_objects.update(keras_resnet.custom_objects)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return resnet_retinanet(*args, backbone=self.backbone, **kwargs)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['resnet50', 'resnet101', 'resnet152']
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return utils.preprocess_image(inputs, mode='custom_tf')

# Function returns the classification sub-model to be used at each feature pyramid level
def default_classification_model(num_classes, num_anchors, pyramid_feature_size=256, prior_probability=0.01, classification_feature_size=256,name='classification_submodel'):
    """ Creates the default regression submodel.
    Args
        num_classes                 : Number of classes to predict a score for at each feature level.
        num_anchors                 : Number of anchors to predict classification scores for at each feature level.
        pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
        prior_probability           : For initialising (See paper)
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name                        : The name of the submodel.
    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

# Function returns the regression sub-model to be used at each feature pyramid level
def default_regression_model(num_values, num_anchors, pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):
    """ Creates the default regression submodel.
    Args
        num_values              : Number of values to regress.
        num_anchors             : Number of anchors to regress for each feature level.
        pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
        regression_feature_size : The number of filters to use in the layers in the regression submodel.
        name                    : The name of the submodel.
    Returns
        A keras.models.Model that predicts regression values for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * num_values, name='pyramid_regression', **options)(outputs)
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

# Function returns [P3, P4, P5, P6, P7] <- outputs of the feature pyramid module of the retinaNet
def __create_pyramid_features(C3, C4, C5, feature_size=256):
    """ [VERIFIED] Creates the FPN layers on top of the backbone features. 'Stage' in the network denotes all the layers which
    outputs the feature maps of same size. And we choose the last layer output of each stage for pyramid feature since
    it is semantically richer than other layers of the stage. SEE PAPER "FPN FOR OBJECT DETECTION", 2017 for more details.
    Args:
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.
    Returns:
        A list of feature levels [P3, P4, P5, P6, P7]. P6 and P7 are added for large object detections
    """
    # upsample C5 to get P5 from the FPN paper
    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4) # to decrease the depth
    P4 = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return [P3, P4, P5, P6, P7]

# Function applies a single sub-model to each FPN level
def __build_model_pyramid(name, model, features):
    """ Applies a single submodel to each FPN level.
    Args
        name     : Name of the submodel.
        model    : The submodel to evaluate.
        features : The FPN features.
    Returns
        A tensor containing the response from the submodel on the FPN features.
    """
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])

# Functions applies all su-bmodels to each FPN level
def __build_pyramid(models, features):
    """ Applies all submodels to each FPN level.
    Args
        models   : List of sumodels to run on each pyramid level (by default only regression, classifcation).
        features : The FPN features.
    Returns
        A list of tensors, one for each submodel.
    """
    return [__build_model_pyramid(name, model, features) for name, model in models]


# Create the rest of the RetinaNet model on the top of a backbone
def retinanet(inputs, backbone_layers, num_classes, num_anchors= None, create_pyramid_features=__create_pyramid_features, submodels = None, name = 'retinanet'):
    """ Construct a RetinaNet model on top of a backbone.
    This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).
    Args:
        inputs                  : keras.layers.Input (or list of) for the input to the model.
        num_classes             : Number of classes to classify.
        num_anchors             : Number of base anchors.
        create_pyramid_features : Functor for creating pyramid features given the features C3, C4, C5 from the backbone.
        submodels               : Submodels to run on each feature map (default is regression and classification submodels).
        name                    : Name of the model.
    Returns:
        A keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.
        The order of the outputs is as defined in sub-models:
        ```
        [
            regression, classification, other[0], other[1], ...
        ]
        ```
    """

    if num_anchors is None: # Return number of anchors i.e. len(self.ratios) * len(self.scales)
        num_anchors = AnchorParameters_default.num_anchors()

    if submodels is None: # Return a list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
        submodels = [('regression', default_regression_model(4, num_anchors)),
                      ('classification', default_classification_model(num_classes, num_anchors))]

    C3, C4, C5 = backbone_layers

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    features = create_pyramid_features(C3, C4, C5)

    # for all pyramid levels, run available submodels
    pyramids = __build_pyramid(submodels, features)

    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


# Get anchor BBox Coords in image space (x1,y1,x2,y2) per anchor_type(=9), per feature_map cell(=feat_h*feat_w), per feature pyramid level(=5), per image in the batch(=1)
def __build_anchors(anchor_parameters, features):
    """ Builds anchors for the shape of the features from FPN.
    Args
        anchor_parameters : Parameteres that determine how anchors are generated.
        features          : The FPN features.
    Returns
        A tensor containing the anchors for the FPN features.
        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    """
    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)


def retinanet_bbox(model= None, applyNms= True, class_specific_filter= True, name='retinanet-bbox', anchor_params= None, **kwargs):

    """ Construct a RetinaNet model on top of a backbone and adds convenience functions to output boxes directly.
    This model uses the minimum retinanet model and appends a few layers to compute boxes within the graph.
    These layers include applying the regression values to the anchors and performing NMS.
    Args:
        model                 : RetinaNet model to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
        nms                   : Whether to use non-maximum suppression for the filtering step.
        class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
        name                  : Name of the model.
        anchor_params         : Struct containing anchor parameters. If None, default values are used.
        *kwargs               : Additional kwargs to pass to the minimal retinanet model.
    Returns:
        A keras.models.Model which takes an image as input and outputs the detections on the image.
        The order is defined as follows:
        ```
        [
            boxes, scores, labels, other[0], other[1], ...
        ]
        ```
    """

    # if no anchor parameters are passed, use default values
    if anchor_params is None:
        anchor_params = AnchorParameters_default

    # create RetinaNet model
    if model is None:
        model = retinanet(num_anchors=anchor_params.num_anchors(), **kwargs)
    else:
        utils.assert_training_model(model)

    # compute the anchors
    features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
    anchors  = __build_anchors(anchor_params, features)

    # we expect the anchors, regression and classification values as first output
    regression     = model.outputs[0]
    classification = model.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    other = model.outputs[2:]

    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # - filter detections (apply NMS / score threshold / select top-k)
    # - nms_threshold used 0.5(default), score_threshold = 0.05(min), max_detections = 300
    # - detections = [boxes, scores, labels] + other_
    detections = layers.FilterDetections(
        nms                   = applyNms,
        class_specific_filter = class_specific_filter,
        name                  = 'filtered_detections'
    )([boxes, classification] + other)

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=detections, name=name)


# Create the complete retinaNet model with the chosen resnet backbone
def resnet_retinanet(num_classes, backbone='resnet50', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a resnet backbone.
    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('resnet50', 'resnet101', 'resnet152')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).
    Returns
        RetinaNet model with a ResNet backbone.
    """
    # choose default input
    if inputs is None:
        if keras.backend.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(3, None, None))
        else:
            inputs = keras.layers.Input(shape=(None, None, 3))

    # create the resnet backbone
    if backbone == 'resnet50':
        resnet = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet101':
        resnet = keras_resnet.models.ResNet101(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet152':
        resnet = keras_resnet.models.ResNet152(inputs, include_top=False, freeze_bn=True)
    else:
        raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))

    # invoke modifier if given
    if modifier:
        resnet = modifier(resnet)

    # create the full model
    return retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=resnet.outputs[1:], **kwargs)


"""
Choose the specific Resnet model for the backbone out of 'resnet50', 'resnet101' or 'resnet152'
"""
def resnet50_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone='resnet50', inputs=inputs, **kwargs)

def resnet101_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone='resnet101', inputs=inputs, **kwargs)

def resnet152_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone='resnet152', inputs=inputs, **kwargs)