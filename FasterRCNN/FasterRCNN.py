import numpy as np
import os
import pandas as pd
import cv2
import Models
from Shapes import Box, GroundTruthBox, Image
from Parameters import anchor_scales, phases, optimizer, resize_min, resize_max, NUM_ROIS
from util import get_anchors, resize_imgs, create_Prediction_Dict, apply_regr, get_real_coordinates
from ROIUtils import RPNsToROIs, nms
from model_trainer import RpnTrainingManager, train_rpn, combined_rpn_det_trainer, DetTrainingManager
from matplotlib import pyplot as plt
from Losses import cls_loss_rpn, bbreg_loss_rpn
from keras.utils import plot_model

"""
Sources: (1) https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras
        (2) https://github.com/Kelicious/faster_rcnn
"""

network = "vgg16" # choices=('vgg16', 'resnet50', 'resnet101')

def make_image_object(path, codeTesting=False):

    ImageDir = os.path.join(path, 'trainV2')
    CSVFile = os.path.join(path, 'trainV2.csv')
    Images = []

    data = pd.read_csv(CSVFile, skiprows=[0], header=None)
    df = pd.DataFrame({'image_id': data[0], 'xmin': data[1], 'ymin': data[2], 'xmax': data[3], 'ymax': data[4], 'label': data[5]})
    df_g = df.groupby(['image_id'])

    for key in df_g.groups.keys():
        imageName = key
        img = cv2.imread(os.path.join(ImageDir, imageName))
        gt_boxes = []
        for subNum in range(len(df_g.groups[key])):
            index = df_g.groups[key][subNum]
            box = Box(x1=df['xmin'][index], y1=df['ymin'][index], x2=df['xmax'][index], y2=df['ymax'][index])
            gt_box = GroundTruthBox(box=box, obj_cls="Table")
            gt_boxes.append(gt_box)

        # print("Image : {} GT : {}".format(imageName, gt_boxes))
        img_obj = Image(name=imageName, width=img.shape[1], height=img.shape[0], gt_boxes=gt_boxes,
                        image_path=os.path.join(ImageDir, imageName))
        Images.append(img_obj)

        if codeTesting:
            break
    return Images


def train_rpn_det():
    """
        ## NOTE: Make NMS use 2k proposals at train time
        ## NOTE: DEBUGGING Script consisting of all the print statements
    """
    root_dir = os.getcwd()
    path = os.path.join(root_dir, 'data')
    train_images = make_image_object(path, codeTesting=False)
    print("Done making image Objects")

    anchors = get_anchors(anchor_scales)
    anchors_per_loc = len(anchors)
    processed_imgs, resized_ratios = resize_imgs(train_images, min_size=resize_min, max_size=resize_max)
    num_classes = 2
    class_mapping = {'Table': 0, 'bg': 1}

    # Create the record.csv file to record losses, acc and mAP
    record_df = pd.DataFrame(columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls',
                 'loss_class_regr', 'curr_loss', 'mAP'])

    preprocess_func = Models.vgg_preprocess
    get_conv_rows_cols_func = Models.vgg_get_conv_rows_cols
    stride = Models.VGG_Stride

    # Working with VGG only. RPN Model: input=Input(shape=(None, None, 3)), outputs=[x_class, x_regr, base_model.output]
    base_model = Models.vgg16_base(weight_regularizer=Models.WEIGHT_REGULARIZER, bias_regularizer=Models.BIAS_REGULARIZER)
    rpn_model = Models.vgg16_rpn(base_model, include_conv=False, weight_regularizer=Models.WEIGHT_REGULARIZER,
                                 bias_regularizer=Models.BIAS_REGULARIZER, anchors_per_loc=anchors_per_loc)

    # Detector Model: inputs=[base_model.input, roi_input], outputs=[out_class, out_reg]
    detector_base = Models.vgg16_base(weight_regularizer=Models.WEIGHT_REGULARIZER, bias_regularizer=Models.BIAS_REGULARIZER)
    detector_model = Models.vgg16_classifier(NUM_ROIS, num_classes, detector_base,
                                          weight_regularizer=Models.WEIGHT_REGULARIZER,
                                          bias_regularizer=Models.BIAS_REGULARIZER)

    # # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    # img_input = Input(shape=(None, None, 3))
    # roi_input = Input(shape=(None, 4), name='roi_input')
    # model_all = Model([img_input, roi_input], rpn_model.output[:2] + detector_model.output)

    rpn_save_weights_dest = "models/combinedTraining_rpn_weights_{}.h5".format(network)
    det_save_weights_dest = "models/combinedTraining_detector_weights_{}.h5".format(network)
    rpn_save_model_dest = "models/combinedTraining_rpn_model_{}.h5".format(network)
    det_save_model_dest = "models/combinedTraining_detector_model_{}.h5".format(network)
    record_path = "models/record.csv"

    rpn_training_manager = RpnTrainingManager(get_conv_rows_cols_func, stride, preprocess_func=preprocess_func,
                                          anchor_dims=anchors)
    det_training_manager = DetTrainingManager(rpn_model=rpn_model, class_mapping=class_mapping,
                                          preprocess_func=preprocess_func, num_rois=NUM_ROIS, stride=stride, anchor_dims=anchors)

    rpn_model, detector_model = combined_rpn_det_trainer(rpn_model, detector_model, processed_imgs, rpn_training_manager, det_training_manager,
                                                         optimizer=optimizer, phases=phases, save_frequency=2000,
                                                         rpn_save_weights_dest=rpn_save_weights_dest,
                                                         det_save_weights_dest=det_save_weights_dest, recordCSV = record_df, record_path=record_path)

    # # Weights corresponding to minimum loss already getting saved in combined_rpn_det_trainer
    # rpn_model.save_weights(rpn_save_weights_dest)
    # print('Saved {} RPN weights to {}'.format(args.network, rpn_save_weights_dest))
    # detector_model.save_weights(det_save_weights_dest)
    # print('Saved {} DET weights to {}'.format(args.network, det_save_weights_dest))

    rpn_model.save(rpn_save_model_dest)
    print('Saved {} RPN model to {}'.format(network, rpn_save_model_dest))
    detector_model.save(det_save_model_dest)
    print('Saved {} DET model to {}'.format(network, det_save_model_dest))
    print("\n Training Complete.")

    print("Plotting Losses")
    plotLosses(record_path, r_epochs=40)


def test_rpn_det():

    anchors = get_anchors(anchor_scales)
    anchors_per_loc = len(anchors)
    root_dir = os.getcwd()
    test_path = os.path.join(root_dir, 'data', 'test')
    num_classes = 2
    bbox_threshold = 0.7

    # Switch key value for class mapping
    class_mapping = {'Table': 0, 'bg': 1}
    class_mapping = {v: k for k, v in class_mapping.items()}
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

    rpn_save_weights_dest = "models/combinedTraining_rpn_weights_{}.h5".format(network)
    det_save_weights_dest = "models/combinedTraining_detector_weights_{}.h5".format(network)
    preprocess_func = Models.vgg_preprocess
    stride = Models.VGG_Stride

    base_model = Models.vgg16_base(weight_regularizer=Models.WEIGHT_REGULARIZER,
                                   bias_regularizer=Models.BIAS_REGULARIZER)
    rpn_model = Models.vgg16_rpn(base_model, include_conv=False, weight_regularizer=Models.WEIGHT_REGULARIZER,
                                 bias_regularizer=Models.BIAS_REGULARIZER, anchors_per_loc=anchors_per_loc)
    detector_base = Models.vgg16_base(weight_regularizer=Models.WEIGHT_REGULARIZER,
                                      bias_regularizer=Models.BIAS_REGULARIZER)
    detector_model = Models.vgg16_classifier(NUM_ROIS, num_classes, detector_base,
                                             weight_regularizer=Models.WEIGHT_REGULARIZER,
                                             bias_regularizer=Models.BIAS_REGULARIZER)
    print("RPN model I/P shape {} \nand O/P shape {}".format(rpn_model.inputs, rpn_model.output))
    print("Detector model I/P shape {} \nand O/P shape {}".format(detector_model.inputs, detector_model.outputs))
    print("Loading weights")
    rpn_model.load_weights(rpn_save_weights_dest)
    detector_model.load_weights(det_save_weights_dest)
    rpn_model.compile(optimizer='sgd', loss='mse')
    detector_model.compile(optimizer='sgd', loss='mse')

    print("----------------- RPN MODEL -----------------")
    # print(rpn_model.summary())
    # plot_model(rpn_model, to_file='rpnModel.png', show_shapes=True)

    print("----------------- DET MODEL -----------------")
    # print(detector_model.summary())
    # plot_model(detector_model, to_file="detModel.png", show_shapes=True)

    test_images = [image for image in os.listdir(test_path) if image.endswith(".png")]
    for imgName in test_images:
        print(imgName)
        img = cv2.imread(os.path.join(test_path, imgName))
        img_obj = Image(name=imgName, width=img.shape[1], height=img.shape[0], gt_boxes=[],
                        image_path=os.path.join(test_path, imgName))
        resized_img_obj, resized_ratio = img_obj.resize_within_bounds(min_size=600, max_size=1000)
        batched_img = np.expand_dims(preprocess_func(resized_img_obj.data), axis=0)

        # [Y1, Y2] = rpn_model.predict_on_batch(batched_img)

        # Get bboxes by applying NMS
        # R.shape = (300, 4)
        R, rois_prob = RPNsToROIs(rpn_model, batched_img, anchors, stride=stride)

        # convert from (x1,y1,x2,y2) to (x1,y1,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // NUM_ROIS + 1): # comes out to 4 + 1

            ROIs = np.expand_dims(R[NUM_ROIS * jk : NUM_ROIS * (jk + 1), :], axis=0)

            if ROIs.shape[1] == 0:
                break               # For Images on which RPN returns zero ROIs

            if jk == R.shape[0] // NUM_ROIS:
                # padding R if the last set ROIS is less than the required NUM_ROIS,
                # Reqd. since DET network uses fc layer with input size (64,7,7,512)
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], NUM_ROIS, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            print("ROIs: {} having shape = {}".format("ROIs", ROIs.shape))
            print("Batched Image: {}".format(batched_img.shape))

            [P_cls, P_regr] = detector_model.predict_on_batch([batched_img, ROIs]) # P_cls.shape = (1,64,2) and P_regr.shape = (1,64,4)

            # Calculate all classes' b-boxes coordinates on re-sized image
            # Drop 'bg' classes b-boxes
            for ii in range(P_cls.shape[1]):

                # Ignore 'bg' class
                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1): # i.e. if index containing max prob is == shape(=2 for 2 class) - 1
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                    (x, y, w, h) = ROIs[0, ii, :]
                    cls_num = np.argmax(P_cls[0, ii, :])

                    try:
                        (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                        x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                    except:
                        pass

                    bboxes[cls_name].append([stride * x, stride * y, stride * (x + w), stride * (y + h)])
                    probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []

        for key in bboxes:
            bbox = np.array(bboxes[key])
            new_boxes, new_probs = nms(bbox, np.array(probs[key]), overlap_thresh=0.2)

            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]
                # Calculate real coordinates on original image
                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(resized_ratio, x1, y1, x2, y2)

                cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                              (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 4)
                textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
                all_dets.append((key, 100 * new_probs[jk]))
                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                textOrg = (real_x1, real_y1 - 0)

                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 1)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        print(all_dets)
        plt.figure(figsize=(10, 10))
        plt.grid()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()


def train_rpn_step1():
    root_dir = os.getcwd()
    path = os.path.join(root_dir, 'data')
    train_images  = make_image_object(path, codeTesting=False)
    print("Done making image Objects")

    anchors = get_anchors(anchor_scales)
    anchors_per_loc = len(anchors)
    processed_imgs, resized_ratios = resize_imgs(train_images, min_size=resize_min, max_size=resize_max)
    stride, get_conv_rows_cols_func, preprocess_func, rpn_model = None, None, None, None

    if network == "vgg16":
        base_model = Models.vgg16_base(weight_regularizer=Models.WEIGHT_REGULARIZER, bias_regularizer=Models.BIAS_REGULARIZER)
        rpn_model = Models.vgg16_rpn(base_model, weight_regularizer=Models.WEIGHT_REGULARIZER,
                                  bias_regularizer=Models.BIAS_REGULARIZER, anchors_per_loc=anchors_per_loc)
        preprocess_func = Models.vgg_preprocess
        get_conv_rows_cols_func = Models.vgg_get_conv_rows_cols
        stride = Models.VGG_Stride

    elif network == "resnet50":
        base_model = Models.resnet50_base(weight_regularizer=Models.WEIGHT_REGULARIZER,
                                          bias_regularizer=Models.BIAS_REGULARIZER)
        rpn_model = Models.resnet50_rpn(base_model, weight_regularizer=Models.WEIGHT_REGULARIZER,
                                        bias_regularizer=Models.BIAS_REGULARIZER, anchors_per_loc=anchors_per_loc)
        preprocess_func = Models.resnet50_preprocess
        get_conv_rows_cols_func = Models.resnet50_get_conv_rows_cols
        stride = Models.ResNet_Stride

    save_weights_dest = "models/rpn_weights_{}_step1.h5".format(network)
    save_model_dest = "models/rpn_model_{}_step1.h5".format(network)
    training_manager = RpnTrainingManager(get_conv_rows_cols_func, stride, preprocess_func=preprocess_func,
                                          anchor_dims=anchors)
    rpn_model = train_rpn(rpn_model, processed_imgs, training_manager, optimizer,
                          phases=phases, save_frequency=2000, save_weights_dest=save_weights_dest,
                          save_model_dest=save_model_dest)

    rpn_model.save_weights(save_weights_dest)
    print('Saved {} rpn weights to {}'.format(network, save_weights_dest))
    rpn_model.save(save_model_dest)
    print('Saved {} rpn model to {}'.format(network, save_model_dest))


def test_rpn_step1():
    """
    ## NOTE: For evaluation, work with diff num of nms proposals at test time such as 100, 300, 1k
    """

    imgName = "1_Page1.png"
    anchors = get_anchors(anchor_scales)
    anchors_per_loc = len(anchors)
    root_dir = os.getcwd()
    test_path = os.path.join(root_dir, 'data')
    img = cv2.imread(os.path.join(test_path, imgName))

    ## For testing on images with no GT (gt_boxes=0)
    img_obj = Image(name=imgName, width=img.shape[1], height=img.shape[0], gt_boxes=[],
                    image_path=os.path.join(test_path, imgName))
    resized_img, resized_ratio = img_obj.resize_within_bounds(min_size=600, max_size=1000)
    rpn_model, preprocess_func = None, None
    if network == "vgg16":
        base_model = Models.vgg16_base(weight_regularizer=Models.WEIGHT_REGULARIZER, bias_regularizer=Models.BIAS_REGULARIZER)
        rpn_model = Models.vgg16_rpn(base_model, weight_regularizer=Models.WEIGHT_REGULARIZER,
                                  bias_regularizer=Models.BIAS_REGULARIZER, anchors_per_loc=anchors_per_loc)
        preprocess_func = Models.vgg_preprocess
        get_conv_rows_cols_func = Models.vgg_get_conv_rows_cols
        stride = Models.VGG_Stride

    elif network == "resnet50":
        base_model = Models.resnet50_base(weight_regularizer=Models.WEIGHT_REGULARIZER,
                                          bias_regularizer=Models.BIAS_REGULARIZER)
        rpn_model = Models.resnet50_rpn(base_model, weight_regularizer=Models.WEIGHT_REGULARIZER,
                                        bias_regularizer=Models.BIAS_REGULARIZER, anchors_per_loc=anchors_per_loc)
        preprocess_func = Models.resnet50_preprocess
        get_conv_rows_cols_func = Models.resnet50_get_conv_rows_cols
        stride = Models.ResNet_Stride

    save_weights_dest = "models/rpn_weights_{}_step1.h5".format(network)
    rpn_model.load_weights(save_weights_dest, by_name=True)
    rpn_model.compile(optimizer=optimizer, loss=[cls_loss_rpn(anchors_per_loc=anchors_per_loc),
                                                 bbreg_loss_rpn(anchors_per_loc=anchors_per_loc)])

    batched_img = np.expand_dims(preprocess_func(resized_img.data), axis=0)

    # [x_class, x_regr] = rpn_model.predict_on_batch(batched_img)

    # Using get_training_input() of train_detector_step2(), writing a custom PredictionToROIs
    rois, rois_prob = RPNsToROIs(rpn_model, batched_img, anchors, stride=stride)
    pred = create_Prediction_Dict(rois, rois_prob)
    print("Region Predictions: {}".format(pred))


def plotLosses(csvPath=None, r_epochs=40):
    record_df = pd.read_csv(csvPath)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, r_epochs), record_df['mean_overlapping_bboxes'], 'r')
    plt.title('mean_overlapping_bboxes')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, r_epochs), record_df['class_acc'], 'r')
    plt.title('class_acc')

    # plt.show()
    plt.savefig("MOB_ACC.png")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_cls'], 'r')
    plt.title('loss_rpn_cls')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_regr'], 'r')
    plt.title('loss_rpn_regr')
    # plt.show()
    plt.savefig("RPN_LOSS.png")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, r_epochs), record_df['loss_class_cls'], 'r')
    plt.title('loss_class_cls')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, r_epochs), record_df['loss_class_regr'], 'r')
    plt.title('loss_class_regr')
    # plt.show()
    plt.savefig("CLASS_LOSS.png")

    plt.plot(np.arange(0, r_epochs), record_df['curr_loss'], 'r')
    plt.title('total_loss')
    # plt.show()
    plt.savefig("Total_Loss.png")


def drawTable():
    imageName = "1_Page1.png"
    root_dir = os.getcwd()
    image_source_dir = os.path.join(root_dir, 'data','imagesV2/')
    image_target_dir = os.path.join(root_dir, 'data')
    image_path = os.path.join(image_source_dir, imageName)
    image = cv2.imread(image_path)
    img_obj = Image(name=imageName, width=image.shape[1], height=image.shape[0], gt_boxes=[],
                    image_path=os.path.join(image_source_dir, imageName))
    resized_img, resized_ratio = img_obj.resize_within_bounds(min_size=600, max_size=1000)

    fname = imageName + "__detections.png"
    resize_img = resized_img.data

    x1, y1, x2, y2 = [ 16, 36, 26, 41]
    cv2.rectangle(resize_img, (x1*16, y1*16), (x2*16, y2*16), (0, 255, 0), 3)



    cv2.imwrite(os.path.join(image_target_dir, fname), resize_img)


if __name__ == "__main__":
    # train_rpn_step1()
    # plotLosses(csvPath="models/record.csv")
    test_rpn_det()