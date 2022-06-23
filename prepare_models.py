from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
import shutil
import datetime

from PARAMETERS import *
from tkinter_dialog_custom import askdirectory
from tkinter_dialog_custom import askopenfilename
from tkinter_dialog_custom import choosefromlist


class Models:
    def __init__(self):
        # Prepare extraction and segmentation models for inference
        print("Model initialization")

    def preapre_extraction_model(self):
        try:
            cfg = get_cfg()
            path_model = askopenfilename(title="Select model.pth file", initialdir=PATH_MODELS_EXTRACTION, filetypes=[("Model files", "*.pth")])
            config_file = self._get_yaml_cfg(os.path.dirname(path_model))
            if config_file != None:
                cfg.merge_from_file(config_file)
            else:
                backbone = self._get_model_backbone_extraction()
                cfg.merge_from_file(model_zoo.get_config_file(backbone))
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(backbone)  # Let training initialize from model zoo
                cfg.OUTPUT_DIR = os.path.dirname(path_model)
                cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
                os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
                cfg.MODEL.WEIGHTS = path_model
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (DEFAULT_THRESH_EXTRACTION)
                cfg.DATASETS.TEST = ("buildings_val",)
            predictor = DefaultPredictor(cfg)
            return predictor
        except FileNotFoundError:
            return None
    
    def _get_model_backbone_extraction(self):
        return choosefromlist(BACKBONES_EXTRACTION, title="Select extraction network backbone")
    
    def _get_model_backbone_segmentation(self):
        return choosefromlist(BACKBONES_SEGMENTATION, title="Select segmentation network backbone")

    def preapre_segmentation_model(self): 
        try:
            cfg = get_cfg()
            path_model = askopenfilename(title="Select model.pth file", initialdir=PATH_MODELS_SEGMENTATION, filetypes=[("Model files", "*.pth")])
            config_file = self._get_yaml_cfg(os.path.dirname(path_model))
            if config_file != None:
                cfg.merge_from_file(config_file)
            else:
                backbone = self._get_model_backbone_segmentation()
                cfg.merge_from_file(model_zoo.get_config_file(backbone))
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(backbone)
                cfg.OUTPUT_DIR =  os.path.dirname(path_model)
                cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
                os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
                cfg.MODEL.WEIGHTS = path_model
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (DEFAULT_THRESH_SEGMENTATION)
                cfg.DATASETS.TEST = ("TCM_test",)
            predictor = DefaultPredictor(cfg)
            return predictor
        except FileNotFoundError:
            return None

    def create_missing_catalogs(self, DATA_FOLDER_PATH):
        # Creates folders used for data saving and processing if sth is missing 
        if not os.path.exists (os.path.join(DATA_FOLDER_PATH,'images')): os.mkdir(os.path.join(DATA_FOLDER_PATH,'images'))
        if not os.path.exists (os.path.join(DATA_FOLDER_PATH,'otsu_tooth')): os.mkdir(os.path.join(DATA_FOLDER_PATH,'otsu_tooth'))
        if not os.path.exists (os.path.join(DATA_FOLDER_PATH,'segmentation')): os.mkdir(os.path.join(DATA_FOLDER_PATH,'segmentation'))
        if not os.path.exists (os.path.join(DATA_FOLDER_PATH,'stepienie_analyze')): os.mkdir(os.path.join(DATA_FOLDER_PATH,'stepienie_analyze'))
        if not os.path.exists (os.path.join(DATA_FOLDER_PATH,'plots')): os.mkdir(os.path.join(DATA_FOLDER_PATH,'plots'))  

    def _get_yaml_cfg(self,directory):
        for file in os.listdir(directory):
            if ".yaml" in file:
                return os.path.join(directory,file)
        return None
