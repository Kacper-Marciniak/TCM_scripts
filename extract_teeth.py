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
PATHES_LIST =  [r'D:\Konrad\TCM_scan\Skany_nowe_pwr\pwr_b_2_20210930_142600',r'D:\Konrad\TCM_scan\Skany_nowe_pwr\pwr_c_3_20211001_133138']
NUM_SAMPLES = 9999
DISPLAY = False

def preapre_extraction_model():
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("TCM_train",)
    cfg.DATASETS.TEST = ("TCM_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 300 #adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = (1000, 1500)
    cfg.SOLVER.GAMMA = 0.05
    cfg.OUTPUT_DIR =  r"D:\Konrad\TCM_scan\model_temp_output"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.TEST.EVAL_PERIOD = 500
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (0.70)
    cfg.DATASETS.TEST = ("buildings_val",)
    predictor = DefaultPredictor(cfg)
    return predictor
def create_missing_catalogs():
    if os.path.exists (data_path + r'\otsu_tooth') == False: os.mkdir(data_path+ r'\otsu_tooth')


extraction_predictor = preapre_extraction_model()

# Iterate over folders and process each image, save results to csv 
for data_path in PATHES_LIST:
    print("Processing:",data_path)
    create_missing_catalogs()
    min_x,min_y,max_x,max_y,name=[],[],[],[],[]
    files = list(os.listdir(data_path + r'\images'))
    print(files[:10])
    for i,image_name in enumerate(files):
        
        if i > NUM_SAMPLES: break
        img_path = data_path + r'\images' +'\\'+ image_name
        im = cv2.imread(img_path)
        outputs = extraction_predictor(im)
    
        if DISPLAY:
            MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = ['tooth']
            v = Visualizer(im[:, :, ::-1],
                        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
                        scale=0.3,
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
