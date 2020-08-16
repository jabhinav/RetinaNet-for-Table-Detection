import numpy as np
import random
import os
import pandas as pd
import time
import cv2
import keras
import warnings
from model import Parameters
from Shapes import Box, GroundTruthBox, Image
from model.utils import preprocess_image, resize_image
from model.transform import adjust_transform_for_image, apply_transform, transform_aabb
from model.anchors import anchor_targets_bbox, anchors_for_shape, guess_shapes, AnchorParameters


def _read_annotations(csv_data_file, image_dir, codeTesting=False):

    ImageDir = image_dir
    CSVFile = csv_data_file

    imagenames = os.listdir(ImageDir)
    imagenames = [img_name for img_name in imagenames if img_name.endswith('.png')]

    Image_objs = []
    start_time = time.time()
    data = pd.read_csv(CSVFile, skiprows=[0], header=None)
    df = pd.DataFrame({'image_id': data[0], 'xmin': data[1], 'ymin': data[2], 'xmax': data[3], 'ymax': data[4], 'label': data[5]})
    df_g = df.groupby(['image_id']) # to collect multiple annotations for a single image

    for key in df_g.groups.keys():
        imageName = key
        if imageName not in imagenames:
            print("WARNING: Image for the entry {} not found in the data directory".format(imageName))
            continue
        else:
            img = cv2.imread(os.path.join(ImageDir, imageName))
            gt_boxes = []
            for subNum in range(len(df_g.groups[key])):
                index = df_g.groups[key][subNum]
                box = Box(x1=df['xmin'][index], y1=df['ymin'][index], x2=df['xmax'][index], y2=df['ymax'][index])
                gt_box = GroundTruthBox(box=box, obj_cls=df['label'][index])
                gt_boxes.append(gt_box)

            # print("Image : {} GT : {}".format(imageName, gt_boxes))
            img_obj = Image(name=imageName, width=img.shape[1], height=img.shape[0], gt_boxes=gt_boxes,
                            image_path=os.path.join(ImageDir, imageName))
            Image_objs.append(img_obj)

            if codeTesting:
                break
    print("{} annotations loaded in {}".format(len(Image_objs), time.time() - start_time))
    return Image_objs


class Generator(keras.utils.Sequence):

    def __init__(self,
            transform_generator=None,
            visual_effect_generator=None,
            batch_size=1,
            group_method='random',  # one of 'none', 'random', 'ratio'
            shuffle_groups=True,
            image_min_side=800,
            image_max_side=1333,
            transform_parameters=None,
            compute_anchor_targets=anchor_targets_bbox,
            compute_shapes=guess_shapes,
            preprocess_image=preprocess_image,
            config=False
    ):
        """ Initialize Generator object.
        Args
            transform_generator    : A generator used to randomly transform images and annotations.
            batch_size             : The size of the batches to generate.
            group_method           : Determines how images are grouped together (defaults to 'ratio', one of ('none', 'random', 'ratio')).
            shuffle_groups         : If True, shuffles the groups each epoch.
            image_min_side         : After resizing the minimum side of an image is equal to image_min_side.
            image_max_side         : If after resizing the maximum side is larger than image_max_side, scales down further so that the max side is equal to image_max_side.
            transform_parameters   : The transform parameters used for data augmentation.
            compute_anchor_targets : Function handler for computing the targets of anchors for an image and its annotations.
            compute_shapes         : Function handler for computing the shapes of the pyramid for a given input.
            preprocess_image       : Function handler for preprocessing an image (scaling / normalizing) for passing through a network.
            config                 : True, if initialize using param from Parameters.py (non-default option for AnchorParameters)
        """
        self.transform_generator = transform_generator
        self.visual_effect_generator = visual_effect_generator

        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        self.image_min_side = image_min_side
        self.image_max_side = image_max_side
        # self.transform_parameters = transform_parameters or TransformParameters()
        self.transform_parameters = transform_parameters
        self.compute_anchor_targets = compute_anchor_targets
        self.compute_shapes = compute_shapes
        self.preprocess_image = preprocess_image
        self.load_parm_from_config = config

        # Define groups which are list of lists of indexes [batch_1, batch_2, batch_3,.....] where batch_1 = [0,1,2,3,....]
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            self.on_epoch_end()



    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def size(self):
        """ Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')

    def has_label(self, label):
        """ Returns True if label is a known label.
        """
        raise NotImplementedError('has_label method not implemented')

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        raise NotImplementedError('has_name method not implemented')

    def name_to_label(self, name):
        """ Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        """ Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')


    def group_images(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                       range(0, len(order), self.batch_size)]


    def load_image_group(self, group):
        """ Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]


    def load_annotations_group(self, group):
        """ Load annotations for all images in group.
        """
        annotations_group = [self.load_annotations(image_index) for image_index in group]
        for annotations in annotations_group:
            assert (isinstance(annotations,dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(type(annotations))
            assert ('labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert ('bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group


    def filter_annotations(self, image_group, annotations_group, group):
        """ Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations for each image
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete entries corresponding to invalid indices from annotations[k] {k='bboxes, 'labels''}
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    annotations['bboxes'][invalid_indices, :]
                ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)
                    # # np.delete returns the output array with entries deleted corresponding to locations specified along axis

        return image_group, annotations_group


    def random_visual_effect_group_entry(self, image, annotations):
        """ Randomly transforms image and annotation.
        """
        visual_effect = next(self.visual_effect_generator)
        # apply visual effect
        image = visual_effect(image)
        return image, annotations


    def random_visual_effect_group(self, image_group, annotations_group):
        """ Randomly apply visual effect on each image.
        """
        assert (len(image_group) == len(annotations_group))

        if self.visual_effect_generator is None:
            # do nothing
            return image_group, annotations_group

        for index in range(len(image_group)):
            # apply effect on a single group entry
            image_group[index], annotations_group[index] = self.random_visual_effect_group_entry(
                image_group[index], annotations_group[index]
            )

        return image_group, annotations_group


    def random_transform_group_entry(self, image, annotations, transform=None):
        """ Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        if transform is not None or self.transform_generator:
            if transform is None:
                transform = adjust_transform_for_image(next(self.transform_generator), image,
                                                       self.transform_parameters.relative_translation)

            # apply transformation to image
            image = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            annotations['bboxes'] = annotations['bboxes'].copy()
            for index in range(annotations['bboxes'].shape[0]):
                annotations['bboxes'][index, :] = transform_aabb(transform, annotations['bboxes'][index, :])

        return image, annotations


    def random_transform_group(self, image_group, annotations_group):
        """ Randomly transforms each image and its annotations.
        """

        assert (len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_transform_group_entry(image_group[index],
                                                                                             annotations_group[index])

        return image_group, annotations_group


    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side. Here image is the numpy array, whereas FasterRCNN
            uses image objects to refer to resized images
        """
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)


    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """
        # preprocess the image
        image = self.preprocess_image(image, mode="custom_tf")

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations
        annotations['bboxes'] *= image_scale

        # convert to the wanted keras floatx
        image = keras.backend.cast_to_floatx(image)

        return image, annotations


    def preprocess_group(self, image_group, annotations_group):
        """ Preprocess each image and its annotations in its group.
        """
        assert (len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index], annotations_group[index] = self.preprocess_group_entry(image_group[index],
                                                                                       annotations_group[index])

        return image_group, annotations_group


    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        if keras.backend.image_data_format() == 'channels_first':
            image_batch = image_batch.transpose((0, 3, 1, 2))

        return image_batch


    def generate_anchors(self, image_shape):
        anchor_params = None # Set as None to use default values present as AnchorParameters_default in anchors.py
        if self.load_parm_from_config:

            anchor_params = AnchorParameters(
                sizes   = Parameters.sizes,
                strides = Parameters.strides,
                ratios  = Parameters.ratios,
                scales  = Parameters.scales,
            )

        return anchors_for_shape(image_shape, anchor_params=anchor_params, shapes_callback=self.compute_shapes)


    def compute_targets(self, image_group, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        # For the max image shape, generate anchors in numpy format of shape (N, 4)) for all pyramid levels (N spans all pyramid levels)
        anchors   = self.generate_anchors(max_shape)

        batches = self.compute_anchor_targets(
            anchors,
            image_group,
            annotations_group,
            self.num_classes(),
            negative_overlap=Parameters.negative_overlap,
            positive_overlap=Parameters.positive_overlap
        )

        return list(batches)


    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images(not their objects) and annotations
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly apply visual effect
        # image_group, annotations_group = self.random_visual_effect_group(image_group, annotations_group)

        # randomly transform data
        image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        # perform pre-processing steps (scaling, and re-sizing)
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets


    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        inputs, targets = self.compute_input_output(group)

        return inputs, targets


class CSVGenerator(Generator):

    """ Generate data for a custom CSV dataset.
    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """
    def __init__(self, csv_data_file, image_dir, class_mapping, **kwargs):
        """
        :param csv_data_file: Path to the CSV annotations file.
        :param image_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
        """
        self.image_names = []
        self.image_data = []
        self.image_dir = image_dir

        # Take base_dir from annotations file if not explicitly specified.
        if self.image_dir is None:
            base_dir = os.path.dirname(csv_data_file)
            base_file_name = os.path.basename(csv_data_file)
            self.image_dir = os.path.join(base_dir, os.path.splitext(base_file_name)[0])

        self.classes = class_mapping
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.image_data = _read_annotations(csv_data_file, image_dir) # Image Objects
        self.image_names = [image_obj.name for image_obj in self.image_data]

        super(CSVGenerator, self).__init__(**kwargs)

    def size(self):
        """
        Size of the dataset.
        """
        return len(self.image_data)

    def num_classes(self):
        """
        Number of classes in the dataset. Excluding the bg class.
        +1, since the class ids are beginning from 0
        """
        return max(self.classes.values()) + 1

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_path(self, image_index):
        """ Returns the image path for image_index.
        """
        return self.image_data[image_index].image_path

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        return float(self.image_data[image_index].width) / float(self.image_data[image_index].height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        return self.image_data[image_index].data

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        img_obj = self.image_data[image_index]
        annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}

        for gtBox in img_obj.gt_boxes:
            annotations['labels'] = np.concatenate((annotations['labels'], [self.name_to_label(gtBox.objClass)]))
            annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
                float(gtBox.x1),
                float(gtBox.y1),
                float(gtBox.x2),
                float(gtBox.y2),
            ]]))

        return annotations
