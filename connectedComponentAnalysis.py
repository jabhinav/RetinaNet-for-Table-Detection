import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt


def process_graphical_lines(image, image_name):


    if len(image.shape) != 2:
        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    cv2.imwrite(os.path.join("Temp", "{}_0.Orig.jpg".format(image_name)), gray)

    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    cv2.imwrite(os.path.join("Temp", "{}_1.binary.jpg".format(image_name)), bw)

    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw)
    vertical = np.copy(bw)

    # [horiz]
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    cv2.imwrite(os.path.join("Temp", "{}_2.1.horizontal.jpg".format(image_name)), horizontal)

    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 30
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    cv2.imwrite(os.path.join("Temp", "{}_2.2.vertical.jpg".format(image_name)), vertical)

    # Inverse vertical image
    # vertical = cv2.bitwise_not(vertical)
    # cv2.imwrite(os.path.join("Temp", "{}_2.vertical.jpg".format(image_name)), vertical)

    for i, img in enumerate([horizontal, vertical]):
        contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        # contours = contours[0] if imutils.is_cv2() else contours[1]
        if contours != None:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)  # The output of cv2.minAreaRect() is ((x, y), (w, h), angle)
                # rectW = rect[1][0]
                # rectH = rect[1][1]

                if w > 0 and h > 0:
                    # print(rectH)
                    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    image[y:y + h, x:x + w] = (255, 255, 255)

    cv2.imwrite(os.path.join("Temp", "{}_3.Final.jpg".format(image_name)), image)

    return image


def process_image(image, image_name):

    CC_heights = []

    # print(image.shape)
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(os.path.join("Temp", "{}_0.Orig.jpg".format(image_name)), gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0.5)
    cv2.imwrite(os.path.join("Temp", "{}_4.blurred.jpg".format(image_name)), blurred)

    kernel = np.ones((1, 4), np.uint8)
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
    cv2.imwrite(os.path.join("Temp", "{}_5.blackhat.jpg".format(image_name)), blackhat)

    thresh, image_thresholded = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join("Temp", "{}_6.thresholded.jpg".format(image_name)), image_thresholded)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 1))
    # to manipulate the orientation of dilution , large x means horizonatally dilating  more,
    # large y means vertically dilating more
    dilated = cv2.dilate(image_thresholded, kernel, iterations=9)

    # kernel_d = np.ones((2, 5), np.uint8)
    # # dilated = cv2.morphologyEx(image_thresholded, cv2.MORPH_DILATE, kernel_d, anchor=(2, 0), iterations=2)
    # dilated = cv2.morphologyEx(image_thresholded, cv2.MORPH_DILATE, kernel_d)
    # dilated = cv2.morphologyEx(dilated, cv2.MORPH_DILATE, kernel_d)
    cv2.imwrite(os.path.join("Temp", "{}_7.dilated.jpg".format(image_name)), dilated)

    kernel_c = np.ones((1, 10), np.uint8)
    # closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_c, anchor=(2, 0), iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_c)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_c)
    cv2.imwrite(os.path.join("Temp", "{}_8.closed.jpg".format(image_name)), closed)

    # kernel = np.ones((23, 37), np.uint8)
    # opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    # cv2.imwrite(os.path.join("Temp", "{}_6.opened.jpg".format(image_name)), opened)

    contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    # contours = contours[0] if imutils.is_cv2() else contours[1]
    if contours != None:
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour) # The output of cv2.minAreaRect() is ((x, y), (w, h), angle)
            # rectW = rect[1][0]
            # rectH = rect[1][1]

            if w > h and h > 0 and w * h > 500:
                # print(rectH)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
                CC_heights.append(h)

    cv2.imwrite(os.path.join("Temp", "{}_9.Final.jpg".format(image_name)), image)

    return CC_heights


# data_to_plot: keys are different images, each key has a list as value consisting of CC_heights of lr and hr image
def drawPDFHist(data_to_plot, dirName, dataset):

    print("Drawing Histogram")
    ncols = 2 # For # channels of the image
    nrows = len(data_to_plot.keys())
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))

    # Lazy counter so we can remove unwanted axes
    counter = 0
    num_bins = 25
    _range = (0, 100)
    bin_size = _range[1]//num_bins
    for i, key in zip(range(nrows), data_to_plot.keys()):
        for j in range(ncols):
            ax = axes[i][j]
            data = data_to_plot[key][j][1]
            y, bins, _ = ax.hist(data, bins=num_bins, range=_range, color='blue', alpha=0.5, label=data_to_plot[key][j][0])

            print("{} image has highest freq for the bin: {} - {}".format(data_to_plot[key][j][0], bins[np.argmax(y)],
                                                                          bins[np.argmax(y)]+bin_size))
            ax.set_xlabel('CC Height')
            ax.set_ylabel('Occurance count')
            # ax.set_ylim([0, 5])
            leg = ax.legend(loc='upper left')
            leg.draw_frame(False)


    plt.savefig(os.path.join(dirName,"{}_Hist_Plot.png".format(dataset)))




def main():
    datasets_dir = os.path.join(os.getcwd(), 'data', 'Final_test')

    # dataset = "isri-ocr-small-lr"
    # dataset = "prog_pdfs_small"
    # dataset = "ProgPDFs6000_72png"
    dataset = "unlv"

    lr_images = []

    if dataset == "isri-ocr-small-lr":
        lr_images = os.listdir(os.path.join(datasets_dir, dataset))
        lr_images = [image for image in lr_images if image.endswith("-100.jpg")]

    elif dataset == "prog_pdfs_small":
        lr_images = os.listdir(os.path.join(datasets_dir, dataset))
        lr_images = [image for image in lr_images if image.endswith("-72.png")]

    elif dataset == "ProgPDFs6000_72png":
        lr_images = os.listdir(os.path.join(datasets_dir, "ProgPDFs6000_72png"))
        lr_images = [image for image in lr_images if image.endswith(".png")]

    elif dataset == "56D" or dataset == "icdar" or dataset == "unlv":
        lr_images = os.listdir(os.path.join(datasets_dir, dataset))
        lr_images = [image for image in lr_images if image.endswith("-72.png")]

    selected_images = random.choices(lr_images, k=3)
    print("Working on Images: {}".format(selected_images))

    data_to_plot = {}

    for i, image_name in enumerate(selected_images):
        lr_img = cv2.imread(os.path.join(datasets_dir, dataset, image_name))
        # print(lr_img.shape)

        # Choose corresponding hr image
        if dataset == "isri-ocr-small-lr":
            hr_img_name = image_name.split(".")[0] + "." + image_name.split(".")[2]
            print("LR - HR Image Pair: {} - {}".format(image_name, hr_img_name))
            hr_img = cv2.imread(os.path.join(datasets_dir, "isri-ocr-small-hr", hr_img_name))

        elif dataset == "prog_pdfs_small":
            hr_img_name = image_name.split("-")[0] + "-300." + image_name.split(".")[1]
            print("LR - HR Image Pair: {} - {}".format(image_name, hr_img_name))
            hr_img = cv2.imread(os.path.join(datasets_dir, "prog_pdfs_small", hr_img_name))

        elif dataset == "ProgPDFs6000_72png":
            hr_img_name = image_name
            print("LR - HR Image Pair: {} - {}".format(image_name, hr_img_name))
            hr_img = cv2.imread(os.path.join(datasets_dir, "ProgPDFs6000_300png", hr_img_name))
            # print(hr_img.shape)

        elif dataset == "56D" or dataset == "icdar" or dataset == "unlv":
            hr_img_name = image_name.split("-")[0] + "-300." + image_name.split(".")[1]
            print("LR - HR Image Pair: {} - {}".format(image_name, hr_img_name))
            hr_img = cv2.imread(os.path.join(datasets_dir, dataset, hr_img_name))

        lr_img = process_graphical_lines(lr_img, image_name)
        hr_img = process_graphical_lines(hr_img, hr_img_name)
        lr_img_data = process_image(lr_img, image_name)
        hr_img_data = process_image(hr_img, hr_img_name)
        data_to_plot[i] = [(image_name, lr_img_data), (hr_img_name, hr_img_data)]

    drawPDFHist(data_to_plot, dirName="Temp", dataset=dataset)


if __name__ == '__main__':
    main()