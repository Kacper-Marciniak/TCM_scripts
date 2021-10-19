import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pandas as pd
import sys
from shutil import copyfile
import random

PATH_LIST = [r'H:\Konrad\Skany_nowe_pwr\pwr_a_1_20210930_100324',r'H:\Konrad\Skany_nowe_pwr\pwr_b_1_20210930_131438',r'H:\Konrad\Skany_nowe_pwr\pwr_c_1_20211001_095602',
            r'H:\Konrad\tcm_scan\20210623_101921',r'H:\Konrad\tcm_scan\20210621_092043',r'H:\Konrad\tcm_scan\20210621_121539',r'H:\Konrad\tcm_scan\20210621_092043']

IMAGES_TO_COPY = [r'H:\Konrad\Skany_nowe_pwr\pwr_a_1_20210930_100324_data']

# ALL_DATA_PATH
ALL_DATA_PATH = r'H:\Konrad\_all_traning_data'
ALL_DATA_SUBFOLDERS = [r'H:\Konrad\_all_traning_data\train',r'H:\Konrad\_all_traning_data\val',r'H:\Konrad\_all_traning_data\train',r'H:\Konrad\_all_traning_data\test',
                      r'H:\Konrad\_all_traning_data\annotations\train',r'H:\Konrad\_all_traning_data\annotations\val',r'H:\Konrad\_all_traning_data\annotations\test']

def recreate_annotation(image_name):
    xml_name = image_name[:image_name.rfind('.')] + '.xml'
    file = open( ML_PATH + r'\annotations\xmls'+ '\\'+ xml_name, "r")
    xmin,ymin,xmax,ymax=0,0,0,0
    for line in file:
        line = line.strip()
        if str(line).find('xmin') != -1: xmin = int(line[ line.rfind('<xmin>') + 6 : line.rfind('</xmin>') ])
        if str(line).find('ymin') != -1: ymin = int(line[ line.rfind('<ymin>') + 6 : line.rfind('</ymin>') ])
        if str(line).find('xmax') != -1: xmax = int(line[ line.rfind('<xmax>') + 6 : line.rfind('</xmax>') ])
        if str(line).find('ymax') != -1: ymax = int(line[ line.rfind('<ymax>') + 6 : line.rfind('</ymax>') ])
    anntoation='''
<annotation>
	<folder>{}</folder>
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
'''.format(subfolder,image_name,ALL_DATA_PATH + '\\'+ subfolder + '\\'+ path + '_tooth_' + str(tooth_base_number + i) + '.png',3840,2748,xmin,ymin,xmax,ymax)   

    a = open(ALL_DATA_PATH + r'\annotations'+'\\' + subfolder + '\\' + path + '_tooth_' + str(tooth_base_number + i) + '.xml', "w")
    a.write(anntoation)
    a.close()

for ML_PATH in IMAGES_TO_COPY:
    print(ML_PATH)
    files = list(os.listdir(ML_PATH + r'\images'))
    print('Images to copy:',len(files))

    # Delete previous files from current catalog to not double images 
    deleted = 0
    path = ML_PATH[ML_PATH.rfind('\\')+1:]
    for folder in ALL_DATA_SUBFOLDERS:
        f = list(os.listdir(folder))
        for data in f:
            if str(data).find(path) != -1:
                deleted +=1
                os.remove(folder +'\\'+ data)
    print("Files deleted: ",deleted)


for ML_PATH in IMAGES_TO_COPY:
    # Copy and split images from current catalog 
    files = list(os.listdir(ML_PATH + r'\images'))
    tooth_base_number  = len(list(os.listdir(ALL_DATA_PATH + r'\val' ))) + len(list(os.listdir(ALL_DATA_PATH + r'\test' ))) + len(list(os.listdir(ALL_DATA_PATH + r'\train' )))   
    print('Existing images:',tooth_base_number) 
    for i,image_name in enumerate(files):

        path = ML_PATH[ML_PATH.rfind('\\')+1:]
        chose_path = random.randint(0, 100)
        subfolder = ''
        
        if chose_path < 60:
            subfolder = 'train'
        elif chose_path < 80:
            subfolder = 'val'
        else:
            subfolder = 'test' 

        print(ML_PATH + r'\images' + '\\' + image_name, " ---------> ", ALL_DATA_PATH + '\\'+ subfolder + '\\'+ path + '_tooth_' + str(tooth_base_number + i) + '.png') 
        copyfile(ML_PATH + r'\images' + '\\' + image_name,ALL_DATA_PATH + '\\'+ subfolder + '\\'+ path + '_tooth_' + str(tooth_base_number + i) + '.png')
        recreate_annotation(image_name)
        cv.waitKey(0)


       
