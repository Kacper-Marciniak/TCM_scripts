from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
import os
import shutil
import datetime

from PARAMETERS import *


class Models:
    def __init__(self):
        # Prepare extraction and segmentation models for inference
        print("Model initialization")

    def preapre_extraction_model(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.OUTPUT_DIR = PATH_TRAINING_OUTPUT_DIR_EXTRACTION
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (0.70)
        cfg.DATASETS.TEST = ("buildings_val",)
        predictor = DefaultPredictor(cfg)
        return predictor
    def preapre_segmentation_model(self): 
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.OUTPUT_DIR =  PATH_TRAINING_OUTPUT_SEGMENTATION
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (0.75)
        cfg.DATASETS.TEST = ("TCM_test",)
        predictor = DefaultPredictor(cfg)
        return predictor

    def create_missing_catalogs(self, DATA_FOLDER_PATH):
        # Creates folders used for data saving and processing if sth is missing 
        if not os.path.exists (os.path.join(DATA_FOLDER_PATH,'images')): os.mkdir(os.path.join(DATA_FOLDER_PATH,'images'))
        if not os.path.exists (os.path.join(DATA_FOLDER_PATH,'otsu_tooth')): os.mkdir(os.path.join(DATA_FOLDER_PATH,'otsu_tooth'))
        if not os.path.exists (os.path.join(DATA_FOLDER_PATH,'segmentation')): os.mkdir(os.path.join(DATA_FOLDER_PATH,'segmentation'))
        if not os.path.exists (os.path.join(DATA_FOLDER_PATH,'stepienie_analyze')): os.mkdir(os.path.join(DATA_FOLDER_PATH,'stepienie_analyze'))
        if not os.path.exists (os.path.join(DATA_FOLDER_PATH,'plots')): os.mkdir(os.path.join(DATA_FOLDER_PATH,'plots'))  

    def copy_and_clear_output_folder(self):
        # Copy files from training output folder to the new folder (separate for each training sequence)
        curr_datetime = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        if len(os.listdir(PATH_TRAINING_OUTPUT_SEGMENTATION))>0:
            path_s = os.path.join(PATH_MODELS_SEGMENTATION, curr_datetime)
            if not os.path.isdir(path_s): os.makedirs(path_s)
            shutil.copytree(PATH_TRAINING_OUTPUT_SEGMENTATION, path_s)
            print(f"Moved all filed from:{PATH_TRAINING_OUTPUT_SEGMENTATION} to: {path_s}")
        if len(os.listdir(PATH_TRAINING_OUTPUT_DIR_EXTRACTION))>0:    
            path_e = os.path.join(PATH_MODELS_EXTRACTION, curr_datetime)
            if not os.path.isdir(path_e): os.makedirs(path_e)
            shutil.copytree(PATH_TRAINING_OUTPUT_DIR_EXTRACTION, path_e)
            print(f"Moved all filed from:{PATH_TRAINING_OUTPUT_SEGMENTATION} to: {path_s}")
        for dir in [path_s,path_e]:
            for file in os.listdir(dir):
                if "model_final" in file: continue # do not remove model_final file
                pth = os.path.join(dir, file)
                try:
                    shutil.rmtree(pth)
                except OSError:
                    os.remove(pth)
                print(f"Removed: {pth}")
