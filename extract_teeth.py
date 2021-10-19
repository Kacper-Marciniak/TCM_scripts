from detectron2.utils.logger import setup_logger
setup_logger()
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
import os
import pandas as pd
import cv2 as cv
import numpy as np

# List of the folders to process
# r'H:\Konrad\tcm_scan\20210621_092043',  ,r'H:\Konrad\Skany_nowe_pwr\pwr_a_2_20210930_104835',r'H:\Konrad\Skany_nowe_pwr\pwr_a_3_20210930_113354'
PATHES_LIST =  [r'H:\Konrad\tcm_scan\20210621_092043',r'H:\Konrad\Skany_nowe_pwr\pwr_a_1_20210930_100324']
NUM_SAMPLES = 9999
DISPLAY = False

# Model loading and configuration
config_file_path = r'C:\Users\Konrad\TCM_scripts\output\config.yml' # from google colab
weights_path = r'C:\Users\Konrad\TCM_scripts\output\model_final.pth' # from google colab
model = config_file_path
cfg = get_cfg()
cfg.merge_from_file(config_file_path)
cfg.MODEL.WEIGHTS = weights_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65
predictor = DefaultPredictor(cfg)

# Iterate over folders and process each image, save results to csv 
for data_path in PATHES_LIST:
    print("Processing:",data_path)
    min_x,min_y,max_x,max_y,name=[],[],[],[],[]
    files = list(os.listdir(data_path + r'\images'))
    print(files[:10])
    for i,image_name in enumerate(files):
        
        if i > NUM_SAMPLES: break
        img_path = data_path + r'\images' +'\\'+ image_name
        im = cv2.imread(img_path)
        outputs = predictor(im)
    
        if DISPLAY:
            MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = ['tooth']
            v = Visualizer(im[:, :, ::-1],
                        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
                        scale = 0.3,
                        instance_mode = ColorMode.SEGMENTATION)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow(image_name,v.get_image()[:, :, ::-1])
            cv2.waitKey(0)
            
        folder_name = data_path[data_path.rfind('\\')+1:]
        try:
            minx,miny,maxx,maxy = list(list(outputs["instances"].to("cpu").pred_boxes)[0].numpy())
            minx,miny,maxx,maxy=int(minx),int(miny),int(maxx),int(maxy)

            roi = im.copy()[miny-50:maxy+50,minx-100:maxx+100]
            print(data_path + r'/otsu_tooth/' + folder_name + '_' + image_name)
            cv.imwrite(data_path + r'/otsu_tooth/' + folder_name + '_' + image_name,roi)   
        except:
            try:     
                roi = im.copy()[miny-50:maxy+50,minx:maxx]
                print(data_path + r'/otsu_tooth/' + folder_name + '_' + image_name)
                cv.imwrite(data_path + r'/otsu_tooth/' + folder_name + '_' + image_name,roi)  
            except:
                print("Error in tooth:",image_name)


print("Finished")
