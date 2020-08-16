import cv2
import os
import time
import argparse
import csv
import pandas as pd

"""
Parameters-
Images Location: root_dir/ImageFolder/*.png
* root_dir : Current working directory
* ImageFolder: Folder present in root_dir containing images to be processed for annotation
* ext : Image file extension (ImageFolder should have all the images belonging to extension-"ext")

Code Usage-
* click and drag to select a bounding box for the table
* Press 's' to save the selection (Creates the corresponding entry in the csv file)
* Press 'r' to reset the image
* Press 'q' to exit the current selection and go on to next one

Code output-
1. generated.csv : Contains the BBox location of chosen tables
2. [OPTIONAL] Cropped tables : Set self.crop = True
"""

root_dir = os.getcwd()
ext = ".png"
ImageFolder = "NitinPageSeg1/"

class annotateBBox:
    def __init__(self, refPt, img, image, img_dir):
        self.refPt = refPt
        self.img = img
        self.image = image
        self.scale = 1
        self.img_dir = img_dir
        self.crop = True

    def click_and_drag(self, event, x, y, flags, param):

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that BBox annotation is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = [(x, y)]

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the BBox annotation is finished
            self.refPt.append((x, y))

            # draw a rectangle around the region of interest
            cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 2)
            cv2.imshow(self.img, self.image)


    def performAnnotation(self, clone, table_entries):
        self.clone = clone

        self.image = cv2.resize(self.image, (0, 0), fx=1 / self.scale, fy=1 / self.scale, interpolation=cv2.INTER_AREA)
        cv2.namedWindow(self.img)
        cv2.setMouseCallback(self.img, self.click_and_drag)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow(self.img, self.image)
            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                self.image = clone.copy()
                self.image = cv2.resize(self.image, (0, 0), fx=1 / self.scale, fy=1 / self.scale, interpolation=cv2.INTER_AREA)

            if key == ord("s"):
                if len(self.refPt) == 2:
                    tableDict = {}
                    tableDict['image_id'] = self.img
                    tableDict['label'] = 'table'
                    tableDict['xmin'] = min(int(self.refPt[0][0]*self.scale), int(self.refPt[1][0]*self.scale))
                    tableDict['ymin'] = min(int(self.refPt[0][1]*self.scale), int(self.refPt[1][1]*self.scale))
                    tableDict['xmax'] = max(int(self.refPt[0][0]*self.scale), int(self.refPt[1][0]*self.scale))
                    tableDict['ymax'] = max(int(self.refPt[0][1]*self.scale), int(self.refPt[1][1]*self.scale))
                    self.refPt = []
                    table_entries.append(tableDict)

                    print("Region Of Interest Saved")
                    roi = self.clone[tableDict['ymin']:tableDict['ymax'], tableDict['xmin']:tableDict['xmax']]

                    if self.crop:
                        cv2.imwrite(os.path.join(self.img_dir , "{}__({},{})_({},{})".format(self.img.replace(ext,""),tableDict['xmin'], tableDict['ymin'], tableDict['xmax'],tableDict['ymax']) + ext), roi)
                else:
                    print("Not enough coordinates to save. Or too many. Exiting")
                    break

            # if the 'q' key is pressed, break from the loop
            elif key == ord("q"):
                cv2.destroyAllWindows()
                break

def main():

    # global refPt, image, img
    table_entries = []
    img_dir = os.path.join(root_dir, ImageFolder)
    image_list = [file for file in os.listdir(img_dir) if file.endswith(ext)]

    for i, img  in enumerate(image_list):
        print("Working on  image {}/{}: {}".format(i+1, len(image_list), img))
        image = cv2.imread(os.path.join(img_dir, img))
        refPt = []
        imageObj = annotateBBox(refPt, img, image, img_dir)
        clone = image.copy()
        imageObj.performAnnotation(clone, table_entries)
    print("Printing annotated Dataset: {}".format(table_entries))

    # print("Creating CSV File..")
    # csv_fields = ['image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
    # CSVFile = 'generated_abhinav_2.csv'
    # with open(CSVFile, 'w') as csvfile:
    #     # creating a csv dict writer object
    #     writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
    #
    #     # writing headers (field names)
    #     writer.writeheader()
    #
    #     # writing data rows
    #     writer.writerows(table_entries)

if __name__ == '__main__':
    main()

## cv2.waitKey(1) & 0xFF
# cv2.waitKey() returns a 32 Bit integer value (might be dependent on the platform).
# The key input is in ASCII which is an 8 Bit integer value.
# So you only care about these 8 bits and want all other bits to be 0.