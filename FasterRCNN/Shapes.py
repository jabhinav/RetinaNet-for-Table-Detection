import cv2
import numpy as np

class Image:
    """
    Encapsulates an image used for network training or inference. This implementation lazily loads the image contents
    from disk to avoid excessive memory usage.
    """

    def __init__(self, name, width, height, gt_boxes, image_path, flipped=False):
        """
        Initializes this image with metadata.
        """
        self._name = name
        self._width = width
        self._height = height
        self._gt_boxes = gt_boxes
        self._image_path = image_path
        self._flipped = flipped

    @property
    def data(self):
        """
        Public attribute to access the image's pixels. This implementation loads the data from disk when called.
        :return: the pixels as a numpy array, resized and/or flipped according to the metadata properties.
        """
        img = cv2.imread(self._image_path)
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        if self.flipped:
            img = cv2.flip(img, 1)

        return img

    @property
    def flipped(self):
        """
        Whether or not to horizontally flip the image when returning its pixels to the user.
        :return: boolean value, true if horizontal flipping is needed.
        """
        return self._flipped

    @property
    def width(self):
        """
        Desired width of the image. Internally the raw pixels are resized to this width.
        :return: positive integer denoting the width in pixels.
        """
        return self._width

    @property
    def height(self):
        """
        Desired height of the image. Internally the raw pixels are resized to this height.
        :return: positive integer denoting the height in pixels.
        """
        return self._height

    @property
    def gt_boxes(self):
        """
        Ground truth boxes for objects in this image. Used for training and evaluation.
        :return: list of GroundTruthBox instances, one for each annotated object in the image.
        """
        return self._gt_boxes

    @property
    def num_gt_boxes(self):
        """
        How many ground truth objects are in this image.
        :return: positive integer.
        """
        return len(self._gt_boxes)

    @property
    def name(self):
        """
        Name of the image, should correspond to the filename without its extension.
        :return: string.
        """
        return self._name

    @property
    def cache_key(self):
        """
        A string guaranteed to be different for different training images, or the same image flipped and not flipped.
        Used to cache training inputs between training iterations of the same input.
        :return: string for caching.
        """
        return self._name + str(self._flipped)

    @property
    def image_path(self):
        "Path to the image on the filesystem as a string."
        return self._image_path

    def resize(self, scale_ratio):
        """
        Resizes the image by a given ratio, maintaining the aspect ratio.
        :param scale_ratio: floating point number by which to multiply the height and width.
        :return: a copy of this object with the new size.
        """
        new_width, new_height = int(round(scale_ratio * self._width)), int(round(scale_ratio * self._height))

        if len(self._gt_boxes) > 0:
            resized_gt_boxes = [gt_box.resize(scale_ratio) for gt_box in self._gt_boxes]
        else:
            resized_gt_boxes = []
        return Image(name = self._name, width=new_width, height=new_height, gt_boxes=resized_gt_boxes, image_path=self._image_path, flipped=self._flipped)

    def resize_within_bounds(self, min_size, max_size):
        """
        Resizes the image to a minimum side length or a maximum side length, whichever is smaller. For example, an image
        with dimensions 200x500 resized with with parameters (400, 1000) will become 400x1000, but if it were resized
        with parameters (400, 750) it would end up as 300x750.
        :param min_size: desired length of the shorter side in pixels.
        :param max_size: maximum length of the longer side in pixels.
        :return: a copy of this object with the new size.
        """
        short_dim = min(self._width, self._height)
        long_dim = max(self._width, self._height)

        min_scale_ratio = min_size / short_dim
        new_max_size = min_scale_ratio * long_dim
        max_scale_ratio = max_size / long_dim

        scale_ratio = max_scale_ratio if new_max_size > max_size else min_scale_ratio
        return self.resize(scale_ratio), scale_ratio

    def horizontal_flip(self):
        """
        Flips the image horizontally.
        :return: a copy of this object flipped horizontally.
        """
        flipped_gt_boxes = [gt_box.horizontal_flip(self._width) for gt_box in self._gt_boxes]

        return Image(name=self._name, width=self._width, height=self._height, gt_boxes=flipped_gt_boxes,
                        image_path=self._image_path, flipped=not self._flipped)


class GroundTruthBox:
    """
    Metadata for ground truth objects in an Image. Used for training and evaluation.
    """
    def __init__(self, box, obj_cls):
        self.box = box
        self.obj_cls = obj_cls

    @property
    def x1(self):
        """
        x coordinate of the top left corner.
        :return: integer.
        """
        return self.box.x1

    @property
    def y1(self):
        """
        y coordinate of the top left corner.
        :return: integer.
        """
        return self.box.y1

    @property
    def x2(self):
        """
        x coordinate of the bottom right corner.
        :return: integer.
        """
        return self.box.x2

    @property
    def y2(self):
        """
        y coordinate of the bottom right corner.
        :return: integer.
        """
        return self.box.y2

    @property
    def width(self):
        """
        The width of the bounding box in pixels.
        :return: positive integer.
        """
        return self.box.width

    @property
    def height(self):
        """
        The height of the bounding box in pixels.
        :return: positive integer.
        """
        return self.box.height

    @property
    def x_center(self):
        """
        x coordinate of the center of the bounding box in pixels.
        :return: integer.
        """
        return self.box.x_center

    @property
    def y_center(self):
        """
        y coordinate of the center of the bounding box in pixels.
        :return: integer.
        """
        return self.box.y_center

    @property
    def corners(self):
        """
        [x1, y1, x2, y2] coordinates of the bounding box.
        :return: numpy array of integers.
        """
        return self.box.corners

    @property
    def corner_dims(self):
        """
        The coordinates of the bounding box's top left corner and its dimensions in the form [x1, y1, w, h].
        :return: numpy array of integers.
        """
        return self.box.corner_dims

    @property
    def center_dims(self):
        """
        The coordinates of the bounding box center and its dimensions in the form [x_center, y_center, w, h].
        :return: numpy array of integers.
        """
        return self.box.center_dims

    def resize(self, scale_ratio):
        """
        Resizes this box.
        :param scale_ratio: floating point number for the resize ratio.
        :return: copy of this object resized.
        """
        return GroundTruthBox(self.box.resize(scale_ratio), self.obj_cls)

    def horizontal_flip(self, width):
        """
        Converts the coordinates to what they would be in the horizontally flipped image.
        :param width: positive integer, the width of the image this box is in.
        :return: copy of this object with its coordinates in the horizontally flipped image.
        """
        new_x1, new_x2 = width - self.x2, width - self.x1
        new_box = Box(new_x1, self.y1, new_x2, self.y2)
        return GroundTruthBox(box=new_box)

    def __repr__(self):
        return "<shapes.GroundTruthBox box: {}>".format(self.box)


class Box:
    @staticmethod
    def from_center_dims_int(x_center, y_center, width, height):
        """
        Factory method to create a Box from the center coordinates and dimensions.
        :param x_center: x coordinate of the center of the box.
        :param y_center: y coordinate of the center of the box.
        :param width: width of the box in pixels.
        :param height: height of the box in pixels.
        :return: new Box instance at the specified position/dimensions.
        """
        x1 = x_center - width // 2
        x2 = x1 + width
        y1 = y_center - height // 2
        y2 = y1 + height

        return Box(x1, y1, x2, y2)

    @staticmethod
    def from_corners(coords):
        """
        Factory method to create a Box from the coordinates of its top left and bottom right corners.
        :param coords: numpy array of coordinates formatted as [x1, y1, x2, y2].
        :return: new Box instance at the specified position.
        """
        return Box(*coords)

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def width(self):
        """
        Width of the box in pixels.
        :return: positive integer.
        """
        return self.x2 - self.x1

    @property
    def height(self):
        """
        Height of the box in pixels.
        :return: positive integer.
        """
        return self.y2 - self.y1

    @property
    def x_center(self):
        """
        x coordinate of the center of the box in pixels.
        :return: floating point number representing the x coordinate.
        """
        return (self.x2 + self.x1) / 2

    @property
    def y_center(self):
        """
        y coordinate of the center of the box in pixels.
        :return: floating point number representing the y coordinate.
        """
        return (self.y1 + self.y2) / 2

    @property
    def corners(self):
        """
        Coordinates of the top left and bottom right corners.
        :return: numpy array in the format [x1, y1, x2, y2].
        """
        return np.array([self.x1, self.y1, self.x2, self.y2])

    @property
    def corner_dims(self):
        """
        Coordinates of the top left corner and the width and height. Useful for plotting.
        :return: numpy array in the format [x1, y1, w, h].
        """
        return np.array([self.x1, self.y1, self.width, self.height])

    @property
    def center_dims(self):
        """
        Coordinates of the center of the box and its dimensions.
        :return: numpy array in the format, [x_center, y_center, width, height]
        """
        return np.array([self.x_center, self.y_center, self.width, self.height])

    def resize(self, scale_ratio):
        """
        A copy of this Box if the image were resized by the scale ratio.
        :param scale_ratio: floating point number, 1.0 for the same size.
        :return: new Box instance with coordinates in the resized image's coordinate space.
        """
        return Box(x1=self.x1 * scale_ratio,
                   y1=self.y1 * scale_ratio,
                   x2=self.x2 * scale_ratio,
                   y2=self.y2 * scale_ratio)

    def __repr__(self):
        return "<image.Box x1: {}, y1: {}, x2: {}, y2: {}>".format(self.x1, self.y1, self.x2, self.y2)