import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pandas as pd
from matplotlib.image import imread
import scipy.misc
from PIL import Image  

  
from detectron2.utils.logger import setup_logger
setup_logger()
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

#r'D:\Konrad\TCM_scan\dash_skany\pwr_b_1_20210930_131438',
#r'D:\Konrad\TCM_scan\Skany_nowe_pwr\pwr_a_1_20210930_100324',
PATHES_LIST =  [r'D:\Konrad\TCM_scan\dash_skany\pwr_a_1_20210930_100324',
                r'D:\Konrad\TCM_scan\dash_skany\pwr_c_odtwarzalnosc_1_ws,             
                r'D:\Konrad\TCM_scan\dash_skany\pwr_c_odtwarzalnosc_2_ws',
                r'D:\Konrad\TCM_scan\dash_skany\pwr_c_odtwarzalnosc_3_ws',
                r'D:\Konrad\TCM_scan\dash_skany\pwr_c_odtwarzalnosc_4_kc',
                r'D:\Konrad\TCM_scan\dash_skany\pwr_c_odtwarzalnosc_5_kc',
                r'D:\Konrad\TCM_scan\dash_skany\pwr_c_odtwarzalnosc_6_kc'] # Foldery do analizy, wpisz ile chesz - zrobi wszystkie jeden po drugim

DASH_PATH = r'D:\Konrad\TCM_scan\dash'

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
    cfg.OUTPUT_DIR =  r"D:\Konrad\TCM_scan\training_extraction\output"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.TEST.EVAL_PERIOD = 500
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (0.70)
    cfg.DATASETS.TEST = ("buildings_val",)
    predictor = DefaultPredictor(cfg)
    return predictor
def preapre_segmentation_model(): 
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("TCM_train")
    cfg.DATASETS.TEST = ("TCM_val",)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1500
    cfg.TEST.EVAL_PERIOD = 25
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.OUTPUT_DIR =  r"D:\Konrad\TCM_scan\traning_segmentation\output"
    # cfg.INPUT.MIN_SIZE_TRAIN = (256,)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (0.70)
    cfg.DATASETS.TEST = ("TCM_test",)
    predictor = DefaultPredictor(cfg)
    return predictor
def create_missing_catalogs(DATA_FOLDER_PATH):
    if os.path.exists (DATA_FOLDER_PATH + r'\images') == False:     os.mkdir(DATA_FOLDER_PATH+ r'\images')
    if os.path.exists (DATA_FOLDER_PATH + r'\otsu_tooth') == False:     os.mkdir(DATA_FOLDER_PATH + r'\otsu_tooth')
    if os.path.exists (DATA_FOLDER_PATH + r'\segmentation') == False:       os.mkdir(DATA_FOLDER_PATH + r'\segmentation')
def decode_segmentation(im, imageName):

    outputs = segmentation_predictor(im)
    base_name = imageName.split('.')[0]
    pred_masks = outputs["instances"].to("cpu").pred_masks.numpy()
    num_instances = pred_masks.shape[0] 
    pred_masks = np.moveaxis(pred_masks, 0, -1)
    pred_masks_instance = []
    output = np.zeros_like(im)
    scores = outputs["instances"].to("cpu").scores.numpy()
    pred_classes = outputs["instances"].to("cpu").pred_classes.numpy()

    for i in range(num_instances):
        out_png_name = data_path + r'\segmentation' + '\\' + base_name + '-' + str(i) + '.png'     
        pred_masks_instance.append(pred_masks[:, :, i:(i+1)])
        output = np.where(pred_masks_instance[0] == True, 255, output)
        im = Image.fromarray(output)
        output = np.zeros_like(im)
        pred_masks_instance = []
        print(out_png_name)
        im.save(out_png_name)
    return num_instances, pred_classes, scores

extraction_predictor = preapre_extraction_model()
segmentation_predictor = preapre_segmentation_model()
for data_path in PATHES_LIST:   # Iterate over folders  
    create_missing_catalogs(data_path)
    files = list(os.listdir(data_path + r'/images'))
    print(" ")
    print("Processing:",data_path)
    print("Number of images:", len(files))
    

    l, w, c_l, c_w = [],[],[],[] #tooth lenght, tooth width, tooth center coordiante - lenght, tooth center coordinate - width
    l_id, w_id, img_name = [],[],[]
    inst_num, scores, inst_id = [],[],[]
    for image_name in files: # Iterate over files
        
        # Basic image data: name, width tooth id, height tooth id
        base_name = image_name[:image_name.rfind('.')]
        split_name = base_name.split('_')

        # Information about the tooth: width, lenght, centre coordinates
        im = cv2.imread(data_path + r'/images/' + image_name)
        print(image_name)
        try:
            outputs = extraction_predictor(im)
            minx, miny, maxx, maxy = list(list(outputs["instances"].to("cpu").pred_boxes)[0].numpy())

            roi = im.copy()[int(miny)-50:int(maxy)+50, int(minx)-200:int(maxx)+200]     
            cv.imwrite(data_path + r'\otsu_tooth' + '\\' + image_name , roi)
            
            try:
                num_instances, pred_class, score = decode_segmentation(roi, image_name)
                l.append(maxy - miny)
                w.append(maxx - minx)
                c_l.append((maxy + miny)/2)
                c_w.append((maxx + minx)/2)
                l_id.append(int(split_name[0]))
                w_id.append(int(split_name[1]))
                img_name.append(str(image_name))
                inst_num.append(str(num_instances))
                scores.append(str(score))
                inst_id.append(str(pred_class))
            except:
                print("Segmentation error in tooth:",image_name)
                
        except:
            print("Extraction error in tooth:",image_name)
    
    data = {'img_name':img_name, 'l_id':l_id, 'w_id':w_id, 'l':l, 'w':w, 'c_l':c_l, 'c_w':c_w ,'inst_num':inst_num,'scores':scores,'inst_id':inst_id}
    CSV_NAME = data_path.split('.')[0]
    CSV_NAME = CSV_NAME[CSV_NAME.rfind('\\') + 1:] + '.csv'
    df = pd.DataFrame(data, columns= ['img_name','l_id', 'w_id','l', 'w', 'c_l', 'c_w','inst_num','scores','inst_id'])  
    df.to_csv (DASH_PATH + '\\' + CSV_NAME, index = False, header=True)
