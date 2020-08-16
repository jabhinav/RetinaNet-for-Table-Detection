import tensorflow as tf
import time
import argparse
import cv2
from csv_generator import CSVGenerator
from model import losses, utils
from model.utils import freeze as freeze_model
from model.defineModel import ResNetBackbone, retinanet_bbox
from model.Parameters import *
from model.customCallbacks import RedirectModel
from model.anchors import make_shapes_callback
from model.transform import random_transform_generator, random_visual_effect_generator, TransformParameters
# import matplotlib.pyplot as plt
# from keras.utils import plot_model


"""
CODE SOURCE: https://github.com/fizyr/keras-retinanet/
"""


"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parser():

    ap = argparse.ArgumentParser()

    #  --------------
    #   Paths
    #  --------------
    ap.add_argument("--base_dir", type=str, default=os.path.join(os.getcwd(),'data'),
                    help="Path to Base Directory containing datasets")
    ap.add_argument("--result_dir", type=str, default=os.path.join(os.getcwd(),'models'),
                    help="Path to Base Directory containing results")
    ap.add_argument("--random_transform", type=bool, default=False,
                    help="Randomly transform image and annotations.")
    return ap


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = learning_rate
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 80:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('\nLearning rate: ', lr)
    return lr


# Call the function to load weights
def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.
    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights=None, multi_gpu=0, freeze_backbone=False, lr=1e-4, config=None):

    """
    Creates three models (model, training_model, prediction_model).
    :param backbone_retinanet: A function to call to create a retinanet model with a given backbone.
    :param num_classes: The number of classes to train.
    :param weights: The weights to load into the model.
    :param multi_gpu: The number of GPUs to use for training.
    :param freeze_backbone: If True, disables learning for the backbone.
    :param lr: Learning Rate for the optimizer
    :param config: Config parameters, None indicates the default configuration.
    :return:
            model            : The base model. This is also the model that is saved in snapshots.
            training_model   : The training model. If multi_gpu=0, this is identical to model.
            prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used which can be changed in Parameters.py)
    anchor_params = None
    num_anchors = None

    # If multiple GPUs are available, create a multi-gpu model
    if multi_gpu > 1:
        print("Working with multi-gpu model")
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier),
                                       weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier),
                                   weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model. An end-to-end Keras model that takes input of the RetinaNet,
    # computes the output and applies it(i.e. final detections)
    # Use default values of retinanet_bbox i.e. applyNms= True,
    # class_specific_filter= True, anchor_params= None (fn later sets it to default)
    prediction_model = retinanet_bbox(model=model)

    # compile model
    training_model.compile(
        loss={
            'regression': losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.Adam(lr=lr, clipnorm=0.001)
    )

    return model, training_model, prediction_model


def create_callbacks(args, model, training_model, prediction_model, validation_generator=None, batch_size=1):
    """ Creates the callbacks to use during training.
        Args
            model: The base model.
            training_model: The model that is used for training.
            prediction_model: The model that should be used for validation.
            validation_generator: The generator for creating validation data.
            args: parseargs args object.
        Returns:
            A list of callbacks used for training.
    """
    callbacks = []

    # Callback : Evaluation(Optional, see from github)

    # Callback : Checkpoint Save.
    # NOTE: If save_weights_only is False(default), full model is saved using (model.save(filepath)).
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        os.path.join(args.result_dir,
            '{backbone}_{{epoch:02d}}.h5'.format(backbone=backbone_name)
        ),
        verbose=1,
        save_best_only=True,
        monitor="val_loss",
        mode='min'
    )
    model_checkpoint_callback = RedirectModel(model_checkpoint_callback, model)
    callbacks.append(model_checkpoint_callback)

    # Callback : EarlyStopping
    earlyStopping_callback = keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        verbose=1,
        mode='min'
    )
    callbacks.append(earlyStopping_callback)

    # Callback : Reduce LR On Plateau
    reduceLRonPlateau_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=4,
        verbose=1,
        mode='min',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0.5e-7
    )
    callbacks.append(reduceLRonPlateau_callback)

    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
    callbacks.append(lr_scheduler)

    csv_logger = keras.callbacks.CSVLogger(os.path.join(args.result_dir, 'EpochsResults.log'))
    callbacks.append(csv_logger)

    return callbacks


def train(args):

    print("RESULT_DIR: {}".format(args.result_dir))
    print("DATA_DIR: {}".format(args.base_dir))

    # transform_generator = random_transform_generator(flip_x_chance=0.5)
    validation_generator = None
    weights = None # Set to None to begin training, else on supplying it will load them into the model

    backbone = ResNetBackbone(backbone_name)

    # [Optional] val_generator
    # [Optional] transform_generator: create random transform generator for augmenting training data, see implementation
    transform_generator, visual_effect_generator = None, None
    if args.random_transform:

        print("Using Transform Generator")
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
        visual_effect_generator = random_visual_effect_generator(
            contrast_range=(0.9, 1.1),
            brightness_range=(-.1, .1),
            hue_range=(-0.05, 0.05),
            saturation_range=(0.95, 1.05)
        )

    # [Optional] To apply transformation to image for data augmentation use transform_parameters and transform_generator
    # where transform_paramters = transform.TransformParameters
    train_generator = CSVGenerator(csv_data_file=os.path.join(args.base_dir,'trainV2.csv'),
                                   image_dir=os.path.join(args.base_dir,'trainV2'),
                                   class_mapping=class_mapping,
                                   batch_size=batch_size,
                                   transform_parameters=TransformParameters(), transform_generator=transform_generator,
                                   visual_effect_generator=visual_effect_generator)

    validation_generator = CSVGenerator(csv_data_file=os.path.join(args.base_dir, 'val.csv'),
                                   image_dir=os.path.join(args.base_dir, 'val'),
                                   class_mapping=class_mapping,
                                   batch_size=batch_size,
                                   transform_parameters= None, transform_generator=None,
                                   visual_effect_generator=None)


    print('Creating model, this may take a second...')
    model, training_model, prediction_model = create_models(
        backbone_retinanet=backbone.retinanet,              # calls the overall resnet_retinanet() function
        num_classes=train_generator.num_classes(),
        weights=weights,
        multi_gpu=multi_gpu,
        freeze_backbone=False,
        lr=learning_rate
    )

    # print(model.summary())
    # plot_model(model, to_file='Model.png', show_shapes=True)
    # plot_model(training_model, to_file='Model_Train.png', show_shapes=True)
    # plot_model(prediction_model, to_file='Model_Prediction.png', show_shapes=True)

    if 'vgg' in backbone_name or 'densenet' in backbone_name:
        train_generator.compute_shapes = make_shapes_callback(model)
        if validation_generator:
            validation_generator.compute_shapes = train_generator.compute_shapes

    # create the callbacks
    print("Creating Callbacks")
    callbacks = create_callbacks(
        args,
        model,
        training_model,
        prediction_model,
        validation_generator
    )

    print("Training the model")
    History = training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        verbose=1,
        callbacks=callbacks,
        workers=num_workers,
        use_multiprocessing=multiprocessing,
        max_queue_size=max_queue_size,
        validation_data=validation_generator,
        shuffle=True
    )

    print("Model Training History: {}".format(History.history))

    #  Save JSON config to disk
    # json_config = training_model.to_json()
    # with open(os.path.join(args.result_dir, 'model_config.json'), 'w') as json_file:
    #     json_file.write(json_config)


"""
models.load_model 
"""


def test():

    def get_session():
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.compat.v1.Session(config=config)

    # use this environment flag to change which GPU to use
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # set the modified tf session as backend in keras
    tf.compat.v1.keras.backend.set_session(get_session())

    # Set the paths
    infer_model_path = os.path.join(root_dir, 'trained_models', 'inferModel.h5')

    # Custom Objects is needed since we have defined custom layers (or other custom objects) in saved models
    if os.path.exists(infer_model_path):
        print("Found the inference model")
        model = keras.models.load_model(filepath=infer_model_path,
                                        custom_objects=ResNetBackbone(backbone_name).custom_objects)
    else:

        model = tf.keras.models.load_model(filepath=trained_model_path,
                                           custom_objects=ResNetBackbone(backbone_name).custom_objects)

        # check if this is indeed a training model
        utils.check_training_model(model)

        # if the model is not converted to an inference model, use and save it. Next time load that
        model = utils.convert_model(model)
        print("Model Converted")

        # save model
        model.save(infer_model_path)

    # print(model.summary())

    # load label to names mapping for visualization purposes
    labels_to_names = {0: 'table'}

    def test_image(src_image_path, orig_image_path, result_dir, image_name):

        image_name_head, image_name_tail = os.path.splitext(image_name)
        image = cv2.imread(src_image_path)
        # copy to draw on(for datasets which do not require distance transform)
        # draw = image.copy()
        draw = cv2.imread(orig_image_path)
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network, NOTE: should be same as preprocessing in preprocess_group_entry()
        # of csv_generator.py
        image = utils.preprocess_image(image, mode="custom_tf")
        image, image_scale = utils.resize_image(image)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)
        # print("Boxes: {}, scores: {}, labels: {}".format(boxes[0], scores[0], labels[0]))

        # correct for image scale i.e. corresponding to its orig size
        boxes /= image_scale

        box_num = 0

        # visualize detections. ASSUMPTION: Scores are present in increasing order.
        for box, score, label in zip(boxes[0], scores[0], labels[0]):  # Choosing [0] since, we have used batch_size=1
            # scores are sorted so we can break, also since labels are from 0, we do not need -1
            if score < 0.6:
                if box_num == 0:
                    cv2.imwrite(
                        os.path.join(result_dir, "detections_cropped", "{}_{}_{}{}".format(image_name_head,
                                                                                           "noDete_minScore-",
                                                                                           score,
                                                                                           image_name_tail)), draw)
                box_num += 1
                break

            # Chooses label color out of a list which has max size=80 to mark 80 distinct colors for each label
            color = utils.label_color(label)

            b = box.astype(int)
            utils.draw_box(draw, b, color=color)

            # # save cropped image and corresponding txt file containing detections (x1, y1, x2, y2)
            cv2.imwrite(os.path.join(result_dir, "detections_cropped", "{}_{}{}".format(image_name_head,
                                                                                        box_num,
                                                                                        image_name_tail)),
                        utils.extract_box(draw, b))

            caption = "{} {:.3f}".format(labels_to_names[label], score)
            utils.draw_caption(draw, b, caption)

            box_num += 1

        cv2.imwrite(os.path.join(result_dir, 'detections_inImage', image_name), draw)

        # plt.figure(figsize=(15, 15))
        # plt.axis('off')
        # plt.imshow(draw)
        # plt.show()
        # plt.savefig(os.path.join(root_dir, 'data', 'test', '9534_001.png'))

    # src_images = os.path.join(root_dir, 'data')
    # orig_images = os.path.join(root_dir, 'data')
    dest_images = os.path.join(root_dir, 'Results', 'sample')
    # images = os.listdir(src_images)
    # images = [image for image in images if image.endswith(".jpg")]
    # images = [images[0]]

    # print("Working on {} images".format(len(images)))
    # for image in images:
    test_image(src_image_path=os.path.join(root_dir, "data", "orig", "sample_0717_023.jpg"),
               orig_image_path=os.path.join(root_dir, "data", "processed", "sample_0717_023_orig.jpg"),
               result_dir=dest_images,
               image_name="sample_0717_023.jpg")


if __name__ == "__main__":

    print("Available GPUs: {}".format(tf.config.list_physical_devices('GPU')))
    args = parser().parse_args()                    # Use vars() to convert args into dictionary

    # train(args)
    test()
