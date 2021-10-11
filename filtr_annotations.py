import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pandas as pd
import sys
from shutil import copyfile

DATA_FOLDER_PATH =  r'H:\Konrad\tcm_scan_2\20210629_090202'
ML_PATH = r'H:\Konrad\tcm_scan_2\20210629_090202_data'


# ML_PATH
ML_PATH_otsu_tresh = ML_PATH + r'\otsu_tresh'
ML_PATH_otsu_tresh_F =  ML_PATH + r'\otsu_tresh_F'
ML_PATH_annotations =  ML_PATH + r'\annotations\xmls' 
ML_PATH_images =  ML_PATH + r'\images' 
ML_PATH_images_F =  ML_PATH + r'\images_F' 
ML_PATH_otsu_tooth = ML_PATH + r'\otsu_tooth'

# BASE_PATH
DATA_FOLDER_PATH_otsu_tresh = DATA_FOLDER_PATH + r'\otsu_tresh'
DATA_FOLDER_PATH_annotations =  DATA_FOLDER_PATH + r'\annotations\xmls' 
DATA_FOLDER_PATH_images =  DATA_FOLDER_PATH + r'\images' 
DATA_FOLDER_PATH_otsu_tooth = DATA_FOLDER_PATH + r'\otsu_tooth'


def replace_in_annotation(old,new,i):
    # opening the file in read mode
    file = open( ML_PATH_annotations + '\\' + 'tooth_' + str(i) +'.xml', "r")
    replacement = ""
    # using the for loop
    for line in file:
        line = line.strip()
        changes = line.replace(old,new)
        replacement = replacement + changes + "\n"

    file.close()
    # opening the file in write mode
    fout = open( ML_PATH_annotations + '\\' + 'tooth_' + str(i) +'.xml', "w")
    fout.write(replacement)
    fout.close()


files = list(os.listdir(DATA_FOLDER_PATH_otsu_tresh))
for i,image_name in enumerate(files):
   
    if(i>=0):   # Chose images id to check
        img = cv.imread(os.path.join(DATA_FOLDER_PATH_otsu_tresh,image_name))
        cv.namedWindow("Classify image",cv.WINDOW_FREERATIO)
        cv.imshow("Classify image",img)
        cv.resizeWindow("Classify image", 1200, 900) 

        xml_name = image_name[:image_name.rfind('.')] + '.xml'

        k = cv.waitKey(0)
        if  k == ord('g'):
            copyfile(DATA_FOLDER_PATH_otsu_tresh + '\\' + image_name, ML_PATH_otsu_tresh + '\\' + 'tooth_' + str(i) +'.png') 
            copyfile(DATA_FOLDER_PATH_otsu_tooth + '\\' + image_name, ML_PATH_otsu_tooth + '\\' + 'tooth_' + str(i) +'.png')  
            copyfile(DATA_FOLDER_PATH_images + '\\' + image_name, ML_PATH_images + '\\' + 'tooth_' + str(i) +'.png')    

            copyfile(DATA_FOLDER_PATH_annotations + '\\' + xml_name, ML_PATH_annotations + '\\' + 'tooth_' + str(i) +'.xml')   
        elif k == ord('f'):
            copyfile(DATA_FOLDER_PATH_otsu_tresh + '\\' + image_name, ML_PATH_otsu_tresh_F + '\\'+ 'tooth_' + str(i) +'.png')   
            copyfile(DATA_FOLDER_PATH_images + '\\' + image_name, ML_PATH_images_F + '\\' + 'tooth_' + str(i) +'.png') 
        elif k == 27:   #ESC
            cv.destroyAllWindows()
            break    
        
        try:
            replace_in_annotation(image_name, 'tooth_' + str(i) + '.png',i)
            replace_in_annotation(DATA_FOLDER_PATH,ML_PATH,i)
            replace_in_annotation('20210621_092043','20210621_092043_data',i)
        except:
            print('No annotation file for wrong bl')
        
        
        print("Examined {} files, last file: {} ".format(i,image_name))   


