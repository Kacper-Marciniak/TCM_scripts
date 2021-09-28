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

DATA_FOLDER_PATH =  r'C:\Users\Konrad\tcm_scan\20210621_092043'
ML_PATH = r'C:\Users\Konrad\tcm_scan\20210621_092043_data\images'

DATA_FOLDER_PATH_images =  DATA_FOLDER_PATH + r'\images'
DATA_FOLDER_PATH_annotations =  DATA_FOLDER_PATH + r'\annotations\xmls' 
DATA_FOLDER_PATH_otsu_tooth = DATA_FOLDER_PATH + r'\otsu_tooth'
DATA_FOLDER_PATH_otsu_tresh = DATA_FOLDER_PATH + r'\otsu_tresh' 



def create_annotation(img,image_name,xmin,ymin,xmax,ymax):
    path = DATA_FOLDER_PATH + '\\' + image_name
    base_name = image_name[:image_name.rfind('.')]
    xml_name = DATA_FOLDER_PATH_annotations +'\\' + base_name + '.xml'
   
    anntoation='''
<annotation>
	<folder>20210621_092043</folder>
	<filename>{}</filename>
	<path>{}</path>
	<source>
		<database>TCM_database</database>
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
'''.format(image_name,path,img.shape[1],img.shape[0],xmin,ymin,xmax,ymax)   
    print(xml_name)
    #print(anntoation)
    f = open(xml_name, "w")
    f.write(anntoation)
    f.close()

def tresh_otsu(image,image_name):
    
    DATA_FOLDER_OTSU_PATH = DATA_FOLDER_PATH_otsu_tooth + '\\' + image_name
    DATA_FOLDER_TOOTH_PATH = DATA_FOLDER_PATH_otsu_tresh + '\\' + image_name
    
    if(os.path.exists(DATA_FOLDER_OTSU_PATH)): 
        print("File {} exist".format(image_name))
        return -1    # Not overwrite files 

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
    print(minr, minc, maxr, maxc)
    #plt.imshow(image)
    #plt.show()
    create_annotation(image,image_name,minc, minr, maxc, maxr)


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



files = list(os.listdir(DATA_FOLDER_PATH))
for i,image_name in enumerate(files):
    
    if((i>=0 and i<=50) or (i>=1000 and i<=1050) or (i>=2000 and i<=2050) or (i>=2700 and i<=2750)):
        img_path = DATA_FOLDER_PATH +'\\'+ image_name
        img = cv.imread(img_path,-1)
        try:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    
        except:
            print("Image not found")
            sys.exit(1)
        tresh_otsu(img,image_name)
 
    print("Processed images: {}/{}".format(i,len(files)))
    