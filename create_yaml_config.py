from tkinter_dialog_custom import askdirectory, choosefromlist, askopenfilename

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

import os

from PARAMETERS import *

## **********
MODEL_SEGMENTATION = True

if MODEL_SEGMENTATION:
    PATH_TO_MODEL_PTH_FILE = askopenfilename(title="Select model.pth file", initialdir=PATH_MODELS_SEGMENTATION, filetypes=[("Model files", "*.pth")])
else:
    PATH_TO_MODEL_PTH_FILE = askopenfilename(title="Select model.pth file", initialdir=PATH_MODELS_EXTRACTION, filetypes=[("Model files", "*.pth")])
PATH_TO_MODEL_PTH_FILE.replace("\\\\","\\").replace("\\","//")

def get_model_backbone_extraction():
    return choosefromlist(BACKBONES_EXTRACTION, title="Select extraction network backbone")
    
def get_model_backbone_segmentation():
    return choosefromlist(BACKBONES_SEGMENTATION, title="Select segmentation network backbone")

try:
    cfg = get_cfg()
    if MODEL_SEGMENTATION: 
        backbone = get_model_backbone_segmentation()
        cfg.merge_from_file(model_zoo.get_config_file(backbone))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(backbone)
        cfg.OUTPUT_DIR =  os.path.dirname(PATH_TO_MODEL_PTH_FILE)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        cfg.MODEL.WEIGHTS = PATH_TO_MODEL_PTH_FILE
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (DEFAULT_THRESH_SEGMENTATION)
        cfg.DATASETS.TEST = ("buildings_val",)        
    else: 
        backbone = get_model_backbone_extraction()
        cfg.merge_from_file(model_zoo.get_config_file(backbone))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(backbone)  # Let training initialize from model zoo
        cfg.OUTPUT_DIR = os.path.dirname(PATH_TO_MODEL_PTH_FILE)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        cfg.MODEL.WEIGHTS = PATH_TO_MODEL_PTH_FILE
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (DEFAULT_THRESH_EXTRACTION)
        cfg.DATASETS.TEST = ("TCM_test",)
    predictor = DefaultPredictor(cfg)
except FileNotFoundError:
    exit()

with open(os.path.join(cfg.OUTPUT_DIR,"model_final.yaml"), "w") as f:
    f.write(cfg.dump())   # save config to file
    f.close()