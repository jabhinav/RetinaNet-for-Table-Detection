import os
import cv2
import camelot
import csv
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pdf2image import convert_from_path
from shutil import copyfile
import matplotlib
import progressbar
from time import sleep

'''
Datasets with XML GT:

1. icdar2013-competition-dataset-with-gt/competition-dataset-eu
2. icdar2013-competition-dataset-with-gt/competition-dataset-us
3. eu-dataset
4. us-gov-dataset
'''

def combineIcdarDatasetCSVs():
    root_dir = os.getcwd()
    newFolder = "ICDAR2013"
    combinedCSV = "combinedICDAR2013DS.csv"
    Dest = os.path.join(root_dir,newFolder)
    if not os.path.exists(Dest):
        os.mkdir(Dest)

    CSVFilesLocation = os.path.join(root_dir,"Table Detection Datasets")
    ImageFolders = ["competition-dataset-eu", "competition-dataset-us", "eu-dataset", "us-gov-dataset"]
    csv_fields = ['image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
    tableEntries = []

    for num, folder in enumerate(ImageFolders):
        ImagesLocation = os.path.join(CSVFilesLocation, folder)
        CSVFile = "Image-" + folder + ".csv"

        image_list = pd.read_csv(os.path.join(CSVFilesLocation, CSVFile), header=None)[0]
        xmin_list = pd.read_csv(os.path.join(CSVFilesLocation, CSVFile), header=None)[1]
        ymin_list = pd.read_csv(os.path.join(CSVFilesLocation, CSVFile), header=None)[2]
        xmax_list = pd.read_csv(os.path.join(CSVFilesLocation, CSVFile), header=None)[3]
        ymax_list = pd.read_csv(os.path.join(CSVFilesLocation, CSVFile), header=None)[4]

        numItems = len(image_list)
        i = 1
        while i < numItems:
            tableDict = {}
            tableDict['image_id'] = folder + "_" + image_list[i]
            tableDict['xmin'] = int(xmin_list[i])
            tableDict['ymin'] = int(ymax_list[i])
            tableDict['xmax'] = int(xmax_list[i])
            tableDict['ymax'] = int(ymin_list[i])
            tableDict['label'] = 'table'
            tableEntries.append(tableDict)

            copyfile(os.path.join(ImagesLocation, image_list[i]), os.path.join(Dest, tableDict['image_id']))

            i += 1

    with open(combinedCSV, 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(tableEntries)


def PDFCSVtoImageCSV():
    root_dir = os.getcwd()
    path = os.path.join(root_dir, 'Table Detection Datasets')
    SourceCSV = 'PDF-us-gov-dataset.csv'
    TargetCSV = SourceCSV.replace('PDF-','Image-')

    ImageDPI = 500
    PDF_RESOLUTION = 72
    scale = ImageDPI / PDF_RESOLUTION
    csv_fields = ['image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
    tableEntries = []

    image_list = pd.read_csv(os.path.join(path, SourceCSV), header=None)[0]
    xmin_list = pd.read_csv(os.path.join(path, SourceCSV), header=None)[1]
    ymin_list = pd.read_csv(os.path.join(path, SourceCSV), header=None)[2]
    xmax_list = pd.read_csv(os.path.join(path, SourceCSV), header=None)[3]
    ymax_list = pd.read_csv(os.path.join(path, SourceCSV), header=None)[4]

    print("Converting {}".format(SourceCSV))
    numItems = len(image_list)
    i = 1
    while i < numItems:
        tableDict = {}
        tableDict['image_id'] = image_list[i]
        print("Working on {}".format(os.path.join(path, SourceCSV.replace('PDF-','').replace('.csv',''), image_list[i])))
        img = cv2.imread(os.path.join(path, SourceCSV.replace('PDF-','').replace('.csv',''), image_list[i]))
        h = img.shape[0]

        tableDict['xmin'] = int(scale*int(xmin_list[i]))
        tableDict['ymin'] = h - int(scale*int(ymax_list[i]))
        tableDict['xmax'] = int(scale*int(xmax_list[i]))
        tableDict['ymax'] = h - int(scale*int(ymin_list[i]))
        tableDict['label'] = 'table'
        tableEntries.append(tableDict)
        i+=1

    print("Saving {}".format(TargetCSV))
    with open(TargetCSV, 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(tableEntries)

def XMLToCSV():
    root_dir = os.getcwd()
    path = os.path.join(root_dir, 'Table Detection Datasets', 'us-gov-dataset')
    CSVfileName = 'us-gov-dataset.csv'

    csv_fields = ['image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'label']

    XMLFiles = [file for file in os.listdir(path) if file.endswith('reg.xml')]

    tableEntries =[]

    for File in XMLFiles:

        print("Processing {}...".format(File))
        XMLFile = os.path.join(path, File)
        imageName = File.replace('-reg.xml','')

        tree = ET.parse(XMLFile)
        root = tree.getroot()

        for table in root.findall('table'):
            tableDict = {}
            region = table.find('region')
            pageNum = region.attrib['page']
            tableDict['image_id'] = imageName + "_Page" + pageNum + ".png"
            BBox = region.find('bounding-box')
            tableDict['xmin'] = int(BBox.attrib['x1'])
            tableDict['ymin'] = int(BBox.attrib['y1'])
            tableDict['xmax'] = int(BBox.attrib['x2'])
            tableDict['ymax'] = int(BBox.attrib['y2'])
            tableDict['label'] = 'table'

            tableEntries.append(tableDict)
            print("Table Entry: {}".format(tableDict))

    print("Writing to CSV File")
    with open(CSVfileName, 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(tableEntries)

    print("CSV file saved.")


def convertPdfsToImages():
    root_dir = os.getcwd()
    pdf_source_dir = os.path.join(root_dir, 'Table Detection Datasets', 'No-annotation-PDF-TREX-Dataset/')
    PDFs = [pdf for pdf in os.listdir(pdf_source_dir) if pdf.endswith(".pdf") ]
    for i, pdf in enumerate(PDFs):
        print("Converting {}/{}: {}".format(i+1, len(PDFs), pdf))
        pages = convert_from_path(os.path.join(pdf_source_dir, pdf), 500) # DPI: 500
        for i, page in enumerate(pages):
            page.save(os.path.join( pdf_source_dir, pdf.split(".")[0] + "_Page{}.png".format(i+1)), 'PNG')


def preProcessTrainValImages():
    root_dir = os.getcwd()
    file_list = ['trainV2.csv', 'val.csv']
    image_source_dir = os.path.join(root_dir, 'data/imagesV2/')
    data_root = os.path.join(root_dir, 'data')

    for file in file_list:

        image_target_dir = os.path.join(data_root, file.split(".")[0])
        if not os.path.exists(image_target_dir):
            os.mkdir(image_target_dir)

        # read list of image files to process from file
        image_list = pd.read_csv(os.path.join(data_root, file), header=None)[0]

        i=1 # Header is also extracted as the first entry
        bar = progressbar.ProgressBar(maxval=len(image_list)-1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

        print("Start pre-processing images")
        bar.start()
        while i < len(image_list):
            image  = image_list[i]
            target_file = os.path.join(image_target_dir, image)
            # print("Writing target file {}/{} {}".format(i , len(image_list)-1, target_file))

            # open image file
            img = cv2.imread(os.path.join(image_source_dir, image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # perform transformations on image
            b = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5)
            g = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=5)
            r = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=5)

            # merge the transformed channels back to an image
            transformed_image = cv2.merge((b, g, r))
            cv2.imwrite(target_file, transformed_image)

            bar.update(i)
            i+=1
            # sleep(0.1)

        bar.finish()


def preProcessSampleImages():
    root_dir = os.getcwd()
    # image_source_dir = os.path.join(root_dir, 'data/val_POs/')
    image_source_dir = os.path.join(root_dir, 'data','imagesV2/')
    data_root = os.path.join(root_dir, 'data')

    image_target_dir = os.path.join(root_dir, 'data','trainV2/')

    # read list of image files to process from file
    image_list = os.listdir(image_source_dir)
    image_list = [file for file in image_list if file.endswith(".png")]

    print("Start pre-processing images")
    for i, image in enumerate(image_list):

        target_file = os.path.join(image_target_dir, image)

        # open image file
        img = cv2.imread(os.path.join(image_source_dir, image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # perform transformations on image
        b = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5)
        g = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=5)
        r = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=5)

        # merge the transformed channels back to an image
        transformed_image = cv2.merge((b, g, r))
        cv2.imwrite(target_file, transformed_image)

def exportPDFTableDataToCSV():
    root_dir = os.getcwd()
    pdf_file = os.path.join(root_dir,'foo.pdf')
    tables = camelot.read_pdf(pdf_file)
    print(tables)

    ## To export all tables present in the pdf into a csv file
    # tables.export('foo.csv', f='csv', compress=True)

    for i in range(len(tables)):
        print("On Table Num: {}".format(i))
        camelot.plot(tables[i], kind='contour')
        # plt.show()


def drawTables():
    imageName = "1_Page1.png"
    root_dir = os.getcwd()
    image_source_dir = os.path.join(root_dir, 'data','imagesV2/')
    image_path = os.path.join(image_source_dir, imageName)
    image = cv2.imread(image_path)

    fname = imageName + "__detections.png"
    cv2.rectangle(image, (520, 1653), (3736, 3487), (0, 255, 0), 3)

    # h = image.shape[0]
    # scale = 500/72
    # cv2.rectangle(image, (int(77*scale), int(h-155*scale)), (int(520*scale), int(h-246*scale)), (0, 255, 0), 3)


    cv2.imwrite(os.path.join(image_source_dir, fname), image)

if __name__ == '__main__':
    # combineIcdarDatasetCSVs()
    # XMLToCSV()
    # PDFCSVtoImageCSV()
    # preProcessTrainValImages()
    drawTables()
    # exportPDFTableDataToCSV()
    # preProcessSampleImages()
    # convertPdfsToImages()