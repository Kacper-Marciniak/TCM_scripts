import random
import os
from shutil import copyfile
import cv2 as cv

ML_PATH = r'H:\Konrad\tcm_scan_2\20210629_090202_data'

# ML_PATH
ML_PATH_otsu_tresh = ML_PATH + r'\otsu_tresh'
ML_PATH_otsu_tresh_F =  ML_PATH + r'\otsu_tresh_F'
ML_PATH_annotations =  ML_PATH + r'\annotations\xmls' 
ML_PATH_images =  ML_PATH + r'\images' 
ML_PATH_images_F =  ML_PATH + r'\images_F' 
ML_PATH_otsu_tooth = ML_PATH + r'\otsu_tooth'

def replace_in_annotation(old,new,i):
    # opening the file in read mode
    file = open( ML_PATH+ r'\annotations' + '\\' + subfolder + '\\' + image_name, "r")
    replacement = ""
    # using the for loop
    for line in file:
        line = line.strip()
        changes = line.replace(old,new)
        replacement = replacement + changes + "\n"

    file.close()
    # opening the file in write mode
    fout = open( ML_PATH+ r'\annotations' + '\\' + subfolder + '\\' + image_name, "w")
    fout.write(replacement)
    fout.close()

files = list(os.listdir(ML_PATH + r'\annotations\xmls'))
for i,image_name in enumerate(files):
    chose_path=random.randint(0, 100)
    png_name = image_name[:image_name.rfind('.')] + '.png'
    subfolder = ''
    print(image_name,png_name)

    if chose_path < 60:
        subfolder = 'train'
    elif chose_path < 80:
        subfolder = 'val'
    else:
        subfolder = 'test' 
    
    print(subfolder)

    copyfile(ML_PATH + r'\annotations\xmls' + '\\' + image_name, ML_PATH+ r'\annotations' + '\\' + subfolder + '\\' + image_name)
    copyfile(ML_PATH_images + '\\' + png_name , ML_PATH_images + '\\' + subfolder + '\\' + png_name)   
    
    replace_in_annotation('\images','\images\\' + subfolder,i)

    cv.waitKey(0)

files_train = list(os.listdir(ML_PATH + r'\annotations\train'))
print("Train dataset size:",len(files_train))
files_val = list(os.listdir(ML_PATH + r'\annotations\val'))
print("Val dataset size:",len(files_val))
files_test = list(os.listdir(ML_PATH + r'\annotations\test'))
print("Test dataset size:",len(files_test))