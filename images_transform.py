import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pandas as pd
import sys

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from itertools import combinations
import xml.etree.cElementTree as ET

DATA_FOLDER_PATH =  r'C:\Users\Konrad\tcm_scan\20210621_092043\images'
ANNOTATIONS_PATH =  r'C:\Users\Konrad\tcm_scan\20210621_092043\annotations\xmls'


def create_annotation(img,image_name,X,Y):
    path = DATA_FOLDER_PATH + '\\' + image_name
    base_name = image_name[:image_name.rfind('.')]
    xml_name = ANNOTATIONS_PATH +'\\' + base_name + '.xml'
   
    anntoation='''
<annotation>
	<folder>dog_dataset</folder>
	<filename>{}</filename>
	<path>{}</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>{}</width>
		<height>{}</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>tooth</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{}</xmin>
			<ymin>{}</ymin>
			<xmax>{}</xmax>
			<ymax>{}</ymax>
		</bndbox>
	</object>
</annotation>
'''.format(image_name,path,img.shape[0],img.shape[1],X[0],Y[0],X[1],Y[1])   
    print(xml_name)
    #print(anntoation)
    f = open(xml_name, "w")
    f.write(anntoation)
    f.close()

def tresh_otsu(image,image_name):
    DATA_FOLDER_OTSU_PATH = r'C:\Users\Konrad\tcm_scan\20210621_092043\otsu_tresh' + '\\' + image_name
    DATA_FOLDER_TOOTH_PATH = r'C:\Users\Konrad\tcm_scan\20210621_092043\otsu_tooth' + '\\' + image_name
    
    '''if(os.path.exists(DATA_FOLDER_OTSU_PATH)): 
        print("File {} exist".format(image_name))
        return -1    # Not overwrite files '''

    # Apply threshold
    thresh = kapur_threshold(image)
    bw = closing(image > thresh, square(15))

    # Remove artifacts connected to image border
    cleared = clear_border(bw)

    # Label image regions
    label_image = label(cleared)
    # To make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

    max_area_region = 0
    max_region = []
    for region in regionprops(label_image):
        # Find max area region
        if region.area >= max_area_region:
            max_area_region = region.area
            max_region = region
           
    # Draw rectangle around largest region
    minr, minc, maxr, maxc = max_region.bbox
    start = (int(minc),int(minr))
    stop = (int(maxc),int(maxr))
    
    try:
        roi = img.copy()[minr-50:maxr+50,minc-200:maxc+200]
        cv.imwrite(DATA_FOLDER_TOOTH_PATH, roi)
    except:
        print("Wrong blob detected")
    image_label_overlay*=255
    cv.rectangle(image_label_overlay,start,stop,(0,0,255),4)
    cv.imwrite(DATA_FOLDER_OTSU_PATH, image_label_overlay)
    
    create_annotation(image,image_name,X=(minc,maxc),Y=(minr,maxr))

def kapur_threshold(image):
    """ Runs the Kapur's threshold algorithm.
    Reference:
    Kapur, J. N., P. K. Sahoo, and A. K. C.Wong. ‘‘A New Method for Gray-Level
    Picture Thresholding Using the Entropy of the Histogram,’’ Computer Vision,
    Graphics, and Image Processing 29, no. 3 (1985): 273–285.
    @param image: The input image
    @type image: ndarray
    @return: The estimated threshold
    @rtype: int
    """
    hist, _ = np.histogram(image, bins=range(256), density=True)
    c_hist = hist.cumsum()
    c_hist_i = 1.0 - c_hist

    # To avoid invalid operations regarding 0 and negative values.
    c_hist[c_hist <= 0] = 1
    c_hist_i[c_hist_i <= 0] = 1

    c_entropy = (hist * np.log(hist + (hist <= 0))).cumsum()
    b_entropy = -c_entropy / c_hist + np.log(c_hist)

    c_entropy_i = c_entropy[-1] - c_entropy
    f_entropy = -c_entropy_i / c_hist_i + np.log(c_hist_i)

    return np.argmax(b_entropy + f_entropy)
def _get_regions_entropy(hist, c_hist, thresholds):
    """Get the total entropy of regions for a given set of thresholds"""

    total_entropy = 0
    for i in range(len(thresholds) - 1):
        # Thresholds
        t1 = thresholds[i] + 1
        t2 = thresholds[i + 1]

        # print(thresholds, t1, t2)

        # Cumulative histogram
        hc_val = c_hist[t2] - c_hist[t1 - 1]

        # Normalized histogram
        h_val = hist[t1:t2 + 1] / hc_val if hc_val > 0 else 1

        # entropy
        entropy = -(h_val * np.log(h_val + (h_val <= 0))).sum()

        # Updating total entropy
        total_entropy += entropy

    return total_entropy
def _get_thresholds(hist, c_hist, nthrs):
    """Get the thresholds that maximize the entropy of the regions
    @param hist: The normalized histogram of the image
    @type hist: ndarray
    @param c_hist: The cummuative normalized histogram of the image
    @type c_hist: ndarray
    @param nthrs: The number of thresholds
    @type nthrs: int
    """
    # Thresholds combinations
    thr_combinations = combinations(range(255), nthrs)

    max_entropy = 0
    opt_thresholds = None

    # Extending histograms for convenience
    # hist = np.append([0], hist)
    c_hist = np.append(c_hist, [0])

    for thresholds in thr_combinations:
        # Extending thresholds for convenience
        e_thresholds = [-1]
        e_thresholds.extend(thresholds)
        e_thresholds.extend([len(hist) - 1])

        # Computing regions entropy for the current combination of thresholds
        regions_entropy = _get_regions_entropy(hist, c_hist, e_thresholds)

        if regions_entropy > max_entropy:
            max_entropy = regions_entropy
            opt_thresholds = thresholds

    return opt_thresholds
def kapur_multithreshold(image, nthrs):
    """ Runs the Kapur's multi-threshold algorithm.
    Reference:
    Kapur, J. N., P. K. Sahoo, and A. K. C.Wong. ‘‘A New Method for Gray-Level
    Picture Thresholding Using the Entropy of the Histogram,’’ Computer Vision,
    Graphics, and Image Processing 29, no. 3 (1985): 273–285.
    @param image: The input image
    @type image: ndarray
    @param nthrs: The number of thresholds
    @type nthrs: int
    @return: The estimated threshold
    @rtype: int
    """
    # Histogran
    hist, _ = np.histogram(image, bins=range(256), density=True)

    # Cumulative histogram
    c_hist = hist.cumsum()

    return _get_thresholds(hist, c_hist, nthrs)


files = list(os.listdir(DATA_FOLDER_PATH))
for i,image_name in enumerate(files):
    

    if(i>=0):
        img_path = DATA_FOLDER_PATH +'\\'+ image_name
        img = cv.imread(img_path,-1)
        try:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    
        except:
            print("Image not found")
            sys.exit(1)
        tresh_otsu(img,image_name)
 


    print("Processed images: {}/{}".format(i,len(files)))
    