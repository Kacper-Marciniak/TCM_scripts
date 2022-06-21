import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import prepare_models
import shutil

from tkinter_dialog_custom import askdirectory
from PARAMETERS import PATH_EXTRACTED_TEETH_DIR


# Program variables
INPUT_IMG_PATH= r"Z:\Dane TCM\skany 2D\20220510_141503"
#INPUT_IMG_PATH = askdirectory(title="Select folder with images from scanner")     # Path to the folder with the images directly from scaner
OUTPUT_IMG_PATH  = askdirectory(title="Select DASH-SQL server to write to", initialdir=PATH_EXTRACTED_TEETH_DIR)   # Path to the folder with processed images_F
DEBUG = False

models = prepare_models.Models()
extraction_predictor = models.preapre_extraction_model()
if os.path.exists (OUTPUT_IMG_PATH) == False:os.mkdir(OUTPUT_IMG_PATH)

min_x_off, min_y_off, max_x_off, max_y_off = 100,50,100,50 # offseting the ROI bounding box

# Iterate over folders and process each image, save results to csv 
files = list(os.listdir(INPUT_IMG_PATH))
len_list_files = len(files)
for i,image_name in enumerate(files): # Process all available images
    base_name = image_name.split('.')[-2]
    split_name = base_name.split('_')
    row = int(split_name[-1])
    section = int(split_name[-2])
        
    try:
        im_pth = os.path.join(INPUT_IMG_PATH, image_name)
        im = cv.imread(im_pth) # Read image
        outputs = extraction_predictor(im)
        min_x, min_y, max_x, max_y = list(list(outputs["instances"].to("cpu").pred_boxes)[0].numpy())

    except Exception as e:
        print(f"There is a problem with the image: {im_pth}")
        print(f"Exception: {e}")
        try:
            source = im_pth
            destination = OUTPUT_IMG_PATH+"/images/"+image_name
            shutil.copyfile(source, destination)
        except Exception as E:
            print("Can't copy corrupted file")
            print(f"Exception: {E}")

    try: # Extraction
        roi = im.copy()[int(min_y)-min_y_off:int(max_y)+max_y_off, int(min_x)-min_x_off:int(max_x)+min_x_off] # Extracting ROI
        print(f"{image_name} extracted - {i+1}/{len_list_files}")
        if DEBUG:
            cv.imshow("ROI", roi)
            cv.waitKey(0)
        cv.imwrite(f"{OUTPUT_IMG_PATH}/{image_name}" , roi) 
    except Exception as e:
        print("Extraction error:", image_name)
        print(f"Exception: {e}")
print("Finished")
