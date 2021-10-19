import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pandas as pd
import sys
from shutil import copyfile


PATH_LIST = [r'H:\Konrad\Skany_nowe_pwr\pwr_a_1_20210930_100324',r'H:\Konrad\Skany_nowe_pwr\pwr_b_1_20210930_131438',r'H:\Konrad\Skany_nowe_pwr\pwr_c_1_20211001_095602',
            r'H:\Konrad\tcm_scan\20210623_101921',r'H:\Konrad\tcm_scan\20210621_092043',r'H:\Konrad\tcm_scan\20210621_121539']

DATA_FOLDER_PATH =  PATH_LIST[0]

# BASE_PATH
DATA_FOLDER_PATH_otsu_tresh = DATA_FOLDER_PATH + r'\otsu_tresh'
DATA_FOLDER_PATH_annotations =  DATA_FOLDER_PATH + r'\annotations\xmls' 
DATA_FOLDER_PATH_images =  DATA_FOLDER_PATH + r'\images' 
DATA_FOLDER_PATH_otsu_tooth = DATA_FOLDER_PATH + r'\otsu_tooth'

# ML_PATH
ML_PATH = DATA_FOLDER_PATH + '_data'
ML_PATH_otsu_tresh = ML_PATH + r'\otsu_tresh'
ML_PATH_otsu_tresh_F =  ML_PATH + r'\otsu_tresh_F'
ML_PATH_annotations =  ML_PATH + r'\annotations\xmls' 
ML_PATH_images =  ML_PATH + r'\images' 
ML_PATH_images_F =  ML_PATH + r'\images_F' 


ML_LIST = [ML_PATH_otsu_tresh,ML_PATH_otsu_tresh_F,ML_PATH_annotations,ML_PATH_images,ML_PATH_images_F]

def clear_data_dir():
    for d in ML_LIST: 
        files = list(os.listdir(d))
        print("Directory:",d,"Files to be deleted: ",len(files))
        for f in files:
                os.remove(d +'\\'+f)
         

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

def create_missing_catalogs():
    if os.path.exists (ML_PATH) == False: os.mkdir(ML_PATH) 
    if os.path.exists (ML_PATH + r'\annotations') == False: os.mkdir(ML_PATH + r'\annotations')
    for d in ML_LIST: 
        if os.path.exists (d) == False: os.mkdir(d)

files = list(os.listdir(DATA_FOLDER_PATH_otsu_tresh))
create_missing_catalogs()
clear_data_dir()
all_wrong = False
for i,image_name in enumerate(files):
   
    if(i>=0):   # Chose images id to check
        img = cv.imread(os.path.join(DATA_FOLDER_PATH_otsu_tresh,image_name))
        cv.namedWindow("Classify image",cv.WINDOW_FREERATIO)
        cv.imshow("Classify image",img)
        cv.resizeWindow("Classify image", 1200, 900) 

        xml_name = image_name[:image_name.rfind('.')] + '.xml'
        if(all_wrong == False):
            k = cv.waitKey(0)
            if  k == ord('p'): all_wrong = True

        if  k == ord('g'):
            copyfile(DATA_FOLDER_PATH_otsu_tresh + '\\' + image_name, ML_PATH_otsu_tresh + '\\' + 'tooth_' + str(i) +'.png') 
            copyfile(DATA_FOLDER_PATH_images + '\\' + image_name, ML_PATH_images + '\\' + 'tooth_' + str(i) +'.png')    
            copyfile(DATA_FOLDER_PATH_annotations + '\\' + xml_name, ML_PATH_annotations + '\\' + 'tooth_' + str(i) +'.xml')   
        elif k == ord('f') :
            copyfile(DATA_FOLDER_PATH_otsu_tresh + '\\' + image_name, ML_PATH_otsu_tresh_F + '\\'+ 'tooth_' + str(i) +'.png')   
            copyfile(DATA_FOLDER_PATH_images + '\\' + image_name, ML_PATH_images_F + '\\' + 'tooth_' + str(i) +'.png') 
        elif all_wrong == True:
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
            pass
        
        
        print("Examined {} files, last file: {} ".format(i,image_name))   


