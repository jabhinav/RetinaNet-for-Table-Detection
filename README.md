# Table_Detection

Deep Learning Models for Table Detection in PDF Document images:
- RetinaNet (Working Model: Train and Test Functions for the model in RetinaNet.py)
- FasterRCNN (Archived Model: Train and Test Functions for the model in FasterRCNN.py)
- YOLOv3 (Archived Model)

Table Detection requires pre-processing of input image which is using distance transform and saving information provided by EuclideanDistanceTransform, LinearDistanceTransform, MaxDistanceTransform as three channels of the image. Method present in `DetectTablesUtils.py` as `preProcessSampleImages()`

Deep Learning Framework used: Keras with Tensorflow
