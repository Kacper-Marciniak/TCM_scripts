import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pandas as pd
from matplotlib.image import imread
import scipy.misc
from PIL import Image, ImageOps 

  
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
PATHES_LIST =  [
                r'D:\Konrad\TCM_scan\dash_skany\pwr_c_3_20211001_133138',
                r'D:\Konrad\TCM_scan\dash_skany\pwr_a_1_20210930_100324',
                r'D:\Konrad\TCM_scan\dash_skany\pwr_b_1_20210930_131438',
                r'D:\Konrad\TCM_scan\dash_skany\pwr_c_odtwarzalnosc_2_ws',
                ] 

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
    if os.path.exists (DATA_FOLDER_PATH + r'\stepienie_analyze') == False:       os.mkdir(DATA_FOLDER_PATH + r'\stepienie_analyze')
    if os.path.exists (DATA_FOLDER_PATH + r'\plots') == False:       os.mkdir(DATA_FOLDER_PATH + r'\plots')
def decode_segmentation(im, imageName):

    outputs = segmentation_predictor(im)
    base_name = imageName.split('.')[0]
    pred_masks = outputs["instances"].to("cpu").pred_masks.numpy()
    scores = outputs["instances"].to("cpu").scores.numpy()
    pred_classes = outputs["instances"].to("cpu").pred_classes.numpy()
    num_instances = pred_masks.shape[0] 
    pred_masks = np.moveaxis(pred_masks, 0, -1)

    # Save all instances as single masks
    pred_masks_instance = []
    output = np.zeros_like(im)
    for i in range(num_instances):  
        pred_masks_instance.append(pred_masks[:, :, i:(i+1)])
        output = np.where(pred_masks_instance[0] == True, 255, output)
        im = Image.fromarray(output)
        output = np.zeros_like(im)
        pred_masks_instance = []
        out_png_name = data_path + r'\segmentation' + '\\' + base_name + '-' + str(i) + '.png'
        im.save(out_png_name)

    # Combine instances 'stępienie' and save it to the further rows analyzys
    pred_masks_instance_stepienie = []
    output_stepienie = np.zeros_like(im)
    j = 0
    for i in range(num_instances):
        if(pred_classes[i]==2):
            pred_masks_instance_stepienie.append(pred_masks[:, :, i:(i+1)])
            output_stepienie = np.where(pred_masks_instance_stepienie[j] == True, 255, output_stepienie)
            j+=1
    blunt_value = 0
    if(2 in pred_classes):
        im = Image.fromarray(output_stepienie)
        out_png_name = data_path + r'\stepienie_analyze' + '\\' + base_name  + '.png'
        im = ImageOps.grayscale(im)
        im.save(out_png_name)
        img = cv.imread(out_png_name,cv.IMREAD_GRAYSCALE)
        blunt_value = analyze_blunt(img)
    return num_instances, pred_classes, scores, blunt_value
def analyze_blunt(img):
    # Find global bounding box containing all instances
    contours, hierarchy = cv.findContours(img,cv.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0 
    height, width = img.shape
    min_x, min_y = width, height
    max_x = max_y = 0
    for cnt in contours:
        idx += 1
        x,y,w,h = cv.boundingRect(cnt)
        min_x, max_x = min(x, min_x), max(x+w, max_x) # Find global min and max 
        min_y, max_y = min(y, min_y), max(y+h, max_y) # for bounding box  
    return max_y - min_y
def add_instances_categories(df):

    stepienie_lst, narost_lst, zatarcie_lst, wykruszenie_lst = [],[],[],[]
    for IMAGE_NAME in df['img_name']:
        inst_ids = str(df.loc[df['img_name'] == IMAGE_NAME, 'inst_id'])
        inst_ids = inst_ids[inst_ids.rfind('[') + 1:]
        inst_ids = inst_ids[:inst_ids.rfind(']')]
        inst_ids = np.array(inst_ids.split(' '))
        if '2' in inst_ids: stepienie_lst.append(1)
        else: stepienie_lst.append(0)
        if '1' in inst_ids: narost_lst.append(1)
        else: narost_lst.append(0)
        if '3' in inst_ids: zatarcie_lst.append(1)
        else: zatarcie_lst.append(0)     
        if '4'in inst_ids: wykruszenie_lst.append(1) 
        else: wykruszenie_lst.append(0)
    df['stepienie'] = stepienie_lst
    df['narost'] = narost_lst
    df['zatarcie'] = zatarcie_lst
    df['wykruszenie'] = wykruszenie_lst
    return df


##############################################################################################

extraction_predictor = preapre_extraction_model()
segmentation_predictor = preapre_segmentation_model()

for data_path in PATHES_LIST:   # Iterate over folders  
    create_missing_catalogs(data_path)
    files = list(os.listdir(data_path + r'/images'))
    print(" ")
    print("Processing:",data_path)
    print("Number of images:", len(files))

    # Containers for stored values
    l_id, w_id, img_name = [],[],[]
    min_x, min_y, max_x, max_y = [],[],[],[]
    l, w, c_l, c_w = [],[],[],[] 
    inst_num, scores, inst_id, blunt_values = [],[],[],[]

    for image_name in files: # Iterate over files
        base_name = image_name[:image_name.rfind('.')]
        split_name = base_name.split('_')
        row = int(split_name[1])
        
        im = cv2.imread(data_path + r'/images/' + image_name) # Read image
        
        try: # Try tooth extraction  
            print(image_name) 
            # Extracting toooth
            outputs = extraction_predictor(im)
            minx, miny, maxx, maxy = list(list(outputs["instances"].to("cpu").pred_boxes)[0].numpy())
            roi = im.copy()[int(miny)-50:int(maxy)+50, int(minx)-100:int(maxx)+100]     
            cv.imwrite(data_path + r'\otsu_tooth' + '\\' + image_name , roi)
            num_instances, pred_class, score, blunt_value = decode_segmentation(roi, image_name)

            # Preparing data for dataframe
            img_name.append(str(image_name))
            l_id.append(int(split_name[0]))
            w_id.append(int(split_name[1]))
            min_x.append(int(minx))
            min_y.append(int(miny))
            max_x.append(int(maxx)) 
            max_y.append(int(maxy))              
            l.append(maxy - miny)
            w.append(maxx - minx)
            c_l.append((maxy + miny)/2)
            c_w.append((maxx + minx)/2)   
            inst_num.append(str(num_instances))
            scores.append(str(score))
            inst_id.append(str(pred_class))
            blunt_values.append(blunt_value)
         
        except:
            print("Extraction error in tooth:",image_name)
            
        
    print("Saving") 
    print(blunt_values) 
    data = {'img_name':img_name,'minx':min_x,'maxx':max_x,'miny':min_y ,'maxy':max_y,'l_id':l_id, 'w_id':w_id, 'l':l, 'w':w, 'c_l':c_l, 'c_w':c_w, 'inst_num':inst_num,'scores':scores,'inst_id':inst_id,'wielkosc_stepienia':blunt_values}
    CSV_NAME = data_path.split('.')[0]
    CSV_NAME = CSV_NAME[CSV_NAME.rfind('\\') + 1:] + '.csv'
    df = pd.DataFrame(data, columns= ['img_name','minx','maxx','miny','maxy','l_id', 'w_id','l', 'w', 'c_l', 'c_w', 'inst_num','scores','inst_id','wielkosc_stepienia']) 
    df = add_instances_categories(df) 
    df.to_csv (DASH_PATH + '\\' + CSV_NAME, index = False, header=True)



    
