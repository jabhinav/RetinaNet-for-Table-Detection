import argparse
import os
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from collections import Counter
import imutils
import colorsys
import pprint
from matplotlib import pyplot as plt
from kneed import KneeLocator
import time



"""
HSV IMPORTANT INFORMATION
• Hue represents the color type. It can be described in terms of an angle on the above
  circle. Although a circle contains 360 degrees of rotation, the hue value is normalized to
  a range from 0 to 255, with 0 being red.
• Saturation represents the vibrancy of the color. Its value ranges from 0 to 255. The
  lower the saturation value, the more gray is present in the color, causing it to appear
  faded.
• Value represents the brightness of the color. It ranges from 0 to 255, with 0 being
  completely dark and 255 being fully bright.
• White has an HSV value of 0-255, 0-255, 255. Black has an HSV value of 0-255, 0-255, 0. 
  The dominant description for black and white is the term, value. The hue and
  saturation level do not make a difference when value is at max or min intensity level. 
"""
## imutils is the package by adrianpyimagsearch providing image utility functions

def is_grey_scale(img_path):
    img = Image.open(img_path).convert('RGB')
    w,h = img.size
    for i in range(w):
        for j in range(h):
            r,g,b = img.getpixel((i,j))
            if r != g != b: return False
    return True


def parser():

    ap = argparse.ArgumentParser()

    #  --------------
    #   Paths
    #  --------------
    ap.add_argument("--base_dir", type=str, default=os.path.join(os.getcwd(),'data'),
                    help="Path to Base Directory containing datasets")
    ap.add_argument("--result_dir", type=str, default=os.path.join(os.getcwd(),'models'),
                    help="Path to Base Directory containing results")
    ap.add_argument("--image_path", type=str)
    return ap


def suppressWhite(image, hasThresholding):

    white = [255,255,255]
    mask = np.all(image == white, axis=-1)
    image[mask] = [0,0,0]
    hasThresholding = True
    return hasThresholding, image


def rgb_to_hsv_single(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v


def rgb_to_hsv_group(colors):
    hsv_array = np.zeros((colors.shape[0],3))
    for i, color in enumerate(colors):
        [h1,s1,v1] = [int(round(x*255)) for x in colorsys.rgb_to_hsv(color[0]/255., color[1]/255., color[2]/255.)]
        h, s, v = rgb_to_hsv_single(color[0], color[1], color[2])
        # if h1 != h or s1 != s or v1 != v:
        #     print("Lib: {}, custom: {}".format([h1,s1,v1], (h, s, v)))
        # print(h,s,v)
        hsv_array[i] = np.array([h1,s1,v1])
    return hsv_array


def plotColorBar(colorInformation):
    # Create a 500x100 black image
    color_bar = np.zeros((100, 500, 3), dtype="uint8")

    top_x = 0
    for x in colorInformation:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

        color = tuple(map(int, (x['RGB_color'])))

        cv2.rectangle(color_bar, (int(top_x), 0), (int(bottom_x), color_bar.shape[0]), color, -1)
        top_x = bottom_x
    return color_bar


def pretty_print_data(color_info):
  for x in color_info:
    print(pprint.pformat(x))
    print()


def extractPatch(image, lowT = np.array([0,0,0]), highT = np.array([255,255,255])):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV. Since we are converting image into HSV Space, need thresholds in HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threasholds
    lower_threshold = np.array([lowT[0], lowT[1], lowT[2]], dtype=np.uint8)
    upper_threshold = np.array([highT[0], highT[1], highT[2]], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


def extractPatches(image, clusterInfo, dataType):

    useQuantile = True
    for cluster_iter in clusterInfo:
        cluster_index = cluster_iter['cluster_index']
        if useQuantile:
            if dataType == "HSV":
                low = np.clip(cluster_iter['Q1'], 0, 255)
                high = np.clip(cluster_iter['Q3'], 0, 255)
            else:
                low_color = np.array(
                    [int(round(x * 255)) for x in colorsys.rgb_to_hsv(cluster_iter['Q1'][0] / 255.,
                                                                      cluster_iter['Q1'][1] / 255.,
                                                                      cluster_iter['Q1'][2] / 255.)])
                high_color = np.array(
                    [int(round(x * 255)) for x in colorsys.rgb_to_hsv(cluster_iter['Q3'][0] / 255.,
                                                                      cluster_iter['Q3'][1] / 255.,
                                                                      cluster_iter['Q3'][2] / 255.)])

                low = np.clip(low_color, 0, 255)
                high = np.clip(high_color, 0, 255)

        print("Cluster:- Old Index {}: Thresholds Used - Low : {}, High : {}".format(cluster_index, low, high))
        patch = extractPatch(image, low, high)
        cluster_iter['patch'] = patch

    return clusterInfo

"""
The removeBlack function is a utility function to remove out the black pixels and their corresponding cluster. 
Since OpenCV by default doesn't handle transparent images and replaces those with zeros(black in color word).
"""
def removeBlack(estimator_labels, estimator_cluster):
    # Check for black
    hasBlack = False
    deletedIndex = None

    # Get the total number of occurence for each color
    occurance_counter = Counter(estimator_labels)

    print("\nOrig Occurance Counter: {}".format(occurance_counter))

    # Quick lambda function to compare to lists
    compare = lambda x, y: Counter(x) == Counter(y)

    # Loop through the most common occuring colors. If the color of the cluster center matches with [0,0,0], delete it
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int.
        cluster_color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(cluster_color, [0, 0, 0]) == True:

            print("Deleted Color :{}, index:{}".format(cluster_color, x[0]))
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            deletedIndex = x[0]
            break

    print("New Estimator clusters : {}\n".format(estimator_cluster))
    return (occurance_counter, estimator_cluster, hasBlack, deletedIndex)


"""
The getColorInfomation function does all the heavy lifiting to make sense of prediction that came from the clustering.
Taking the prediction labels (estimator_labels) and the cluster centroids(estimator_cluster) as the input and 
returns an array of dictionaries of the extracted colours.
"""
def getColorInformation(estimator_labels, estimator_cluster, RGBcolour_to_label, HSVcolour_to_label, hasThresholding=False, dataType="HSV"):
    # Variable to keep count of the occurance of each color predicted
    occurance_counter = None
    deletedIndex = None


    # Output list variable to return
    colorInformation = []

    # Check for Black
    hasBlack = False

    print("Estimator clusters : {}".format(estimator_cluster))
    # print("Estimator Labels: {}".format(estimator_labels))

    # If a mask has been applied, remove the black
    if hasThresholding == True:

        (occurance, cluster, black, deletedIndex) = removeBlack(estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black

    else:
        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):
        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold. Removing black removes its cluster also.
        # So an index has been removed thus all other indexes has to dec by one
        # index = (index - 1) if ((hasThresholding & hasBlack) & (int(index) != 0)) else index

        if ((hasThresholding & hasBlack) & (int(index) != 0)):
            if index > deletedIndex:
                newIndex = index-1
            else:
                newIndex = index
        else:
            newIndex = index

        # Get the color number into a list. The estimator cluster is already adjusted for the case hasBlack=True
        color = estimator_cluster[newIndex].tolist()

        # Get the percentage of each color
        color_percentage = (x[1] / totalOccurance)

        # make the dictionay of the information
        if dataType == "RGB":
            colorInfo = {'OldIndex': index, "cluster_index": newIndex, "RGB_color": color,
                     "HSV_color": np.array([int(round(x*255)) for x in colorsys.rgb_to_hsv(color[0]/255., color[1]/255., color[2]/255.)]),
                     "color_percentage": color_percentage}
        else:
            colorInfo = {'OldIndex': index, "cluster_index": newIndex, "RGB_color": np.array([int(round(x * 255)) for x in
                                                colorsys.hsv_to_rgb(color[0] / 255., color[1] / 255.,
                                                                    color[2] / 255.)]),
                         "HSV_color": color,
                         "color_percentage": color_percentage}
        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation, deletedIndex



"""
The extractDominantColor is the function that call the above function to output the information.
The function take an 8 bit 3 channel BGR image as the input , the number of colors to be extracted. 
This does all the super heavy lifting by sparkling some magic power of machine learning.
As mention in the article , An unsupervised clustering algorithm, 
KMeans Clustering is used to cluster the pixel data based on their RGB values.
The function also takes an optional parameter (hasThresholding) to indicate whether a thresholding mask was used. 
This passed to the getColorInformation function
"""
"""
Clustering is done in RGB Color space
"""
def extractDominantColor(image, number_of_colors=5, hasThresholding=False, dataType="HSV"):

    # Quick Fix Increase cluster counter to neglect the black(Read Article)
    if hasThresholding == True:
        number_of_colors += 1

    # Convert Image into RGB Colours Space
    RGBimg = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    HSVimg = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)

    # Reshape Image
    RGBimg = RGBimg.reshape((RGBimg.shape[0] * RGBimg.shape[1]), 3)
    HSVimg = HSVimg.reshape((HSVimg.shape[0] * HSVimg.shape[1]), 3)

    ## Perform clustering in which colour space - HSV or RGB
    if dataType == "HSV":
        chosenImg = HSVimg
    else:
        chosenImg = RGBimg

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0, verbose=0)

    # Fit the image
    estimator.fit(chosenImg)


    ## Return RGB colors list grouped by their cluster label
    RGBcolour_to_label = {i: RGBimg[np.where(estimator.labels_ == i)] for i in range(estimator.n_clusters)}
    # HSVcolour_to_label = {i: rgb_to_hsv_group(img[np.where(estimator.labels_ == i)]) for i in range(estimator.n_clusters)}
    HSVcolour_to_label = {i: HSVimg[np.where(estimator.labels_ == i)] for i in range(estimator.n_clusters)}



    # Get Colour Information. .labels_ returns the labels for each entry in "img", .cluster_centers_ consists
    # cluster centre co-ords
    colorInformation , deletedIndex = getColorInformation(estimator.labels_, estimator.cluster_centers_,
                                                          RGBcolour_to_label, HSVcolour_to_label, hasThresholding,
                                                          dataType)
    if deletedIndex is not None:
        del RGBcolour_to_label[deletedIndex]
        del HSVcolour_to_label[deletedIndex]

    return colorInformation, RGBcolour_to_label, HSVcolour_to_label


def drawPDFHist(colour_dict, dataType, dirName):

    n_clusters = len(colour_dict.keys())
    ncols = 3 # For # channels of the image
    nrows = n_clusters
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))

    # Lazy counter so we can remove unwanted axes
    counter = 0
    for i, index in zip(range(nrows), colour_dict.keys()):
        for j in range(ncols):
            ax = axes[i][j]
            data = colour_dict[index][:, j]
            ax.hist(data, bins=50, color='blue', alpha=0.5, label='cluster: {}'.format(index))
            ax.set_xlabel('x')
            ax.set_ylabel('PDF')
            # ax.set_ylim([0, 5])
            leg = ax.legend(loc='upper left')
            leg.draw_frame(False)


    plt.savefig(os.path.join(dirName,"{}_PDF_Plot.png".format(dataType)))


def analyse(colour_dict):
    info = []
    for index in colour_dict.keys():
        color_list = colour_dict[index]
        Cmin = np.min(color_list, axis=0)
        Cmax = np.max(color_list, axis=0)
        Cmean = np.mean(color_list, axis=0).astype(int)
        Cstd = np.std(color_list, axis=0).astype(int)
        Q1 = np.quantile(color_list, 0.25, axis=0)
        Q3 = np.quantile(color_list, 0.75, axis=0)
        Cmedian = np.median(color_list, axis=0)
        perClusterinfo = {"cluster_index": index, "min_value": Cmin, "max_value": Cmax, "Mean":Cmean, "Std":Cstd,
                          "Q1" : Q1, "Q3" : Q3, "Median": Cmedian}
        info.append(perClusterinfo)

    return info


def kneedK(distortions):
    # # https://github.com/arvkevi/kneed
    x = range(1,10)
    y = distortions
    kneedle = KneeLocator(x, y, curve='convex', direction='decreasing')
    print("\nKneed ALgo Optimal K - Knee: {} and Elbow: {}".format(kneedle.knee, kneedle.elbow))
    return kneedle.elbow

def optK(distortions):

    start = time.time()
    distortions = [x/100. for x in distortions]
    delta1 = [distortions[i] - distortions[i+1] for i in range(len(distortions) - 1)]
    delta2 = [delta1[i] - delta1[i+1] for i in range(len(delta1) - 1)]

    delta1.insert(0, None)
    delta2.insert(0, None)
    delta2.insert(0, None)

    strength = [delta2[i+1] - delta1[i+1] for i in range(len(distortions) - 1) if i > 0]
    strength.append(0)
    strength.insert(0, 0)
    strength = [round(x,3) for x in strength]

    # strength = np.array(strength)
    _max = max(strength)
    amax = strength.index(max(strength))

    if _max > 0:
        k_opt = amax + 1
    else:
        k_opt = None

    print("Distortions: {}\n Delta1: {}\n Delta2: {}\n Strength: {}\n K Optimal: {}\n".format(distortions,delta1, delta2, strength, k_opt))
    print("\n----- TIME STAMP -----\n optk fn time: {}".format(round(time.time()-start, 3)))
    return k_opt


def plot_elbow_method(image, dirName):

    # create new plot and data
    plt.figure()
    start = time.time()
    distortions = []
    K = range(1, 10)

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1]), 3)

    HSVimg = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
    HSVimg = HSVimg.reshape((HSVimg.shape[0] * HSVimg.shape[1]), 3)

    chosenImg = HSVimg

    for k in K:
        print(k)
        kmeanModel = KMeans(n_clusters=k, random_state=0, verbose=0)
        kmeanModel.fit(chosenImg)
        distortions.append(sum(np.min(cdist(chosenImg, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / chosenImg.shape[0])

    print("\n----- TIME STAMP -----\n Plot ELBOW fn time: {}".format(round(time.time()-start, 3)))
    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')

    plt.savefig(os.path.join(dirName, "Optimal_K.png"))
    plt.clf()
    plt.cla()
    plt.close()
    return distortions

# # find otsu's threshold value with OpenCV function
# ret, otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


def process_patch(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0.5)

    thresh, image_thresholded = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

def temp_process(args):
    image = cv2.imread(args.image_path)
    HSVimg = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
    bins = 10



"""
Parameters:
Image resize width (same aspect ratio maintained)
number of clusters
"""
def main(i, image_path, image_name):

    dirName = "headerResults/test_{}".format(image_name)
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    image = cv2.imread(image_path)
    cv2.imwrite(os.path.join(dirName, "OrigImage.png"), image)

    # os.chdir("../ClientTestTableDataset/")
    # image = cv2.imread(os.path.join(os.path.abspath(".."),"ClientTestTableDataset/Screenshot 2019-07-29 at 9.17.21 PM.png"))

    # Resize image to a width of 500
    image = imutils.resize(image, width=500)

    hasThresholding = False
    # if is_grey_scale(args.image_path):
    #     hasThresholding = False
    #     print("GRAYSCALE IMAGE")
    # else:
    #     # Suppress White color
    #     hasThresholding, image = suppressWhite(image, hasThresholding)
    #     cv2.imwrite(os.path.join(dirName, "ThresholdedImage.png"), image)

    ## Determine number of optimal clusters
    distortions = plot_elbow_method(image, dirName)
    K_opt_custom = optK(distortions)
    K_opt_kneed = kneedK(distortions)
    numClusters = K_opt_kneed

    dominantColors, RGBcolour_to_label, HSVcolour_to_label = extractDominantColor(image,
                                                                               number_of_colors=numClusters,
                                                                               hasThresholding=hasThresholding)

    # Show in the dominant color information
    print("Color Information")
    pretty_print_data(dominantColors)

    # Show in the dominant color as bar
    print("Color Bar")
    colour_bar = plotColorBar(dominantColors)
    cv2.imwrite(os.path.join(dirName, "colorBar.png"), cv2.cvtColor(colour_bar, cv2.COLOR_RGB2BGR))
    # cv2.imshow("ColorBar",cv2.cvtColor(colour_bar, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

    H_diff = 20
    HSVinfo = analyse(HSVcolour_to_label)
    # print("\nCluster info (HSV Format):")
    # pretty_print_data(HSVinfo)
    drawPDFHist(HSVcolour_to_label, "HSV", dirName)
    modHSVinfo = extractPatches(image, HSVinfo, dataType="HSV")
    for clusterInfo in modHSVinfo:
        if clusterInfo['Q3'][0] - clusterInfo['Q1'][0] < H_diff:
            clusterIsSelected = "Selected"
        else:
            clusterIsSelected = "Not_Selected"
        cv2.imwrite(os.path.join(dirName, '{}_HSVpatch_{}.png'.format(clusterIsSelected, clusterInfo['cluster_index'])),
                    clusterInfo['patch'])

        # process_patch(clusterInfo['patch'])
    #     # cv2.imshow("Patch_{}".format(clusterInfo['cluster_index']), clusterInfo['patch'])
    #     # cv2.waitKey(0)


    # RGBinfo = analyse(RGBcolour_to_label)
    # print("\nCluster info (RGB Format):")
    # pretty_print_data(RGBinfo)
    # drawPDFHist(RGBcolour_to_label, "RGB")
    # modRGBinfo = extractPatches(image, RGBinfo, dataType="RGBToHSV")
    # for clusterInfo in modRGBinfo:
    #     cv2.imwrite('headerResults/RGBpatch_{}.png'.format(clusterInfo['cluster_index']), clusterInfo['patch'])
    #     cv2.imshow("Patch_{}".format(clusterInfo['cluster_index']), clusterInfo['patch'])
    #     cv2.waitKey(0)



    # Convert clustered RGB colors to HSV


if __name__ == "__main__":
    args = parser().parse_args()
    images = os.listdir(args.image_path)
    images = [image for image in images if image.endswith(".png") or image.endswith(".jpg")]
    for i, img in enumerate(images):
        main(i, os.path.join(args.image_path, img), img.split(".")[0])