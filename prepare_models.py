from detectron2.utils.logger import setup_logger
setup_logger()
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
import os


class Models:
    def __init__(self):
        # Prepare extraction and segmentation models for inference
        print("Model initialization")

    def preapre_extraction_model(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.OUTPUT_DIR =  r"D:\Konrad\TCM_scan\training_extraction\output"
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
        cfg.OUTPUT_DIR =  r"D:\Konrad\TCM_scan\traning_segmentation\MODELS\TES-24"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (0.75)
        cfg.DATASETS.TEST = ("TCM_test",)
        predictor = DefaultPredictor(cfg)
        return predictor

    def create_missing_catalogs(self, DATA_FOLDER_PATH):
        # Creates folders used for data saving and processing if sth is missing 
        if os.path.exists (DATA_FOLDER_PATH + r'\images') == False:     os.mkdir(DATA_FOLDER_PATH+ r'\images')
        if os.path.exists (DATA_FOLDER_PATH + r'\otsu_tooth') == False:     os.mkdir(DATA_FOLDER_PATH + r'\otsu_tooth')
        if os.path.exists (DATA_FOLDER_PATH + r'\segmentation') == False:       os.mkdir(DATA_FOLDER_PATH + r'\segmentation')
        if os.path.exists (DATA_FOLDER_PATH + r'\stepienie_analyze') == False:       os.mkdir(DATA_FOLDER_PATH + r'\stepienie_analyze')
        if os.path.exists (DATA_FOLDER_PATH + r'\plots') == False:       os.mkdir(DATA_FOLDER_PATH + r'\plots')  