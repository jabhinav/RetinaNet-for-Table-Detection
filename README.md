# Table Detection

Deep Learning Models for Table Detection in PDF Document images:
- RetinaNet (Working Model: Train and Test Functions for the model in RetinaNet.py)
- FasterRCNN (Archived Model: Train and Test Functions for the model in FasterRCNN.py)
- YOLOv3 (Archived Model)

Table Detection requires pre-processing of input image which is using distance transform and saving information provided by EuclideanDistanceTransform, LinearDistanceTransform, MaxDistanceTransform as three channels of the image. Method present in `DetectTablesUtils.py` as `preProcessSampleImages()`. Loopkup the sample files in `data` folder for the original pdf image and its distance transformed version. 

Deep Learning Framework used: Keras with Tensorflow

**Dataset**: For Dataset of Document Images with Tables present, refer the enclosed `Object Detection Details and Survey`.


