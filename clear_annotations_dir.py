import os
import glob

ML_PATH = r'C:\Users\Konrad\tcm_scan\20210621_092043_data'
ML_PATH_otsu_tresh = ML_PATH + r'\otsu_tresh'
ML_PATH_annotations =  ML_PATH + r'\annotations\xmls' 

DATA_FOLDER_PATH =  r'C:\Users\Konrad\tcm_scan\20210621_092043'
DATA_FOLDER_PATH_otsu_tresh = DATA_FOLDER_PATH + r'\otsu_tresh'
DATA_FOLDER_PATH_annotations =  DATA_FOLDER_PATH + r'\annotations\xmls' 

def clear_dir(dir):
    files = glob.glob(dir)
    for f in files:
        os.remove(f)

clear_dir(DATA_FOLDER_PATH_otsu_tresh)

