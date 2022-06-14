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

# r'\pwr_c_3_20211001_133138',
PATHES_LIST =  [
                r'D:\Konrad\TCM_scan\dash_skany\pwr_c_3_20211001_133138',
                r'D:\Konrad\TCM_scan\dash_skany\pwr_a_1_20210930_100324',
                r'D:\Konrad\TCM_scan\dash_skany\pwr_b_1_20210930_131438',
                r'D:\Konrad\TCM_scan\dash_skany\pwr_c_odtwarzalnosc_2_ws',
                ] # Foldery do analizy, wpisz ile chesz - zrobi wszystkie jeden po drugim


BROACH_DIR = r'D:\Konrad\TCM_scan\dash_skany'  # Path to the corresponding folders with images 
BROACH_CSV = r'D:\Konrad\TCM_scan\dash' # Path to the folder with .csv files


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
def create_missing_catalogs(DATA_FOLDER_PATH):
    if os.path.exists (DATA_FOLDER_PATH + r'\images') == False:     os.mkdir(DATA_FOLDER_PATH+ r'\images')
    if os.path.exists (DATA_FOLDER_PATH + r'\otsu_tooth') == False:     os.mkdir(DATA_FOLDER_PATH + r'\otsu_tooth')
    if os.path.exists (DATA_FOLDER_PATH + r'\segmentation') == False:       os.mkdir(DATA_FOLDER_PATH + r'\segmentation')
    if os.path.exists (DATA_FOLDER_PATH + r'\stepienie_analyze') == False:       os.mkdir(DATA_FOLDER_PATH + r'\stepienie_analyze')
    if os.path.exists (DATA_FOLDER_PATH + r'\plots') == False:       os.mkdir(DATA_FOLDER_PATH + r'\plots')
def max_dim_in_rows(path):
    # Defines boxes for combined teeth in each row
    files = os.listdir(path)
    containers = []
    names = []
    # Find unique rows ids
    for image_name in files: 
        base_name = image_name[:image_name.rfind('.')]
        split_name = base_name.split('_')
        row = split_name[1]
        if int(row) not in names: names.append(int(row))
    # Create table for max x,y images sizes in each row
    for name in names: 
        containers.append((name,3840,2748))
  
    return containers
def fill_pixels_in_blunt(img):
    # Fill bottom part of failure
    contours, hierarchy = cv.findContours(img,cv.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0 
    for cnt in contours: # Iterate over detected contours 
        idx += 1
        x,y,w,h = cv.boundingRect(cnt)
        # roi = img[y:y+h,x:x+w] # Can also utilized for displaying

        for py in range(y+1, y+h): # Change value of the piksel if piksel above it is brighter
            for px in range (x+1, x+w):
                if img[py, px] < img[py-1, px]: img[py, px] = img[py-1, px]

    return img
def normalize_in_x(img):
    for y in range(img.shape[0]): # Change value of the piksel if piksel above it is brighter
        row = 0
        for x in range(img.shape[1]):
            row += img.item(y, x) 
        for x in range(img.shape[1]):
            if (img.item(y, x) > 0): img.itemset(y, x , int(255 - row/img.shape[1]))
    
    return img
def draw_plot(img,name):
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
   
    if idx == 0: return 0 # If no contours brake 

    # Define boxes for cumulative chart 
    bins = [0]*(max_y-min_y)

    roi = img[min_y:max_y,min_x:max_x] #Internal area of the bounding box    
    max_v = 0 # Max piksels in 1 container
    for py in range(0,max_y-min_y):
        for px in range (0,max_x-min_x):
            bins[len(bins)-py-1] += roi[py][px]
        max_v = max(max_v, bins[len(bins)-py-1])
        
    stop = max([i for i,b in enumerate(bins) if b > max_v*0.1]) # Find max bin index with values above 10% 
    start = min([i for i,b in enumerate(bins) if b > max_v*0.1]) # Find max bin index with values above 10% 
    print("start: {} px, stop: {}, dif {:0.3f} mm".format(start,stop,(stop-start)/603))
    plot_name = data_path + r'\plots'+ '\\' + str(name) + '.jpg' 
    plt.plot(bins)
    plt.axhline(y=int(max_v*0.1), color='r', linestyle='-') # 10% line
    plt.axvline(x=start, color='r', linestyle='--') # 10% line
    plt.axvline(x=stop, color='r', linestyle='--') # 10% line
    plt.title("{:0.3f}mm".format((stop-start)/603))
    plt.savefig(plot_name)
    '''
    # Draw for debuging
    cv.normalize(img, img, 0, 255, norm_type = cv.NORM_MINMAX) # Normalize pixels on the image 
    cv2.rectangle(img, (min_x-8, min_y-8), (max_x+8, max_y+8), 255, 4) # Draw bounding box
    cv2.line(img, (min_x,max_y-stop), (max_x,max_y-stop), 255, 2)
    cv2.line(img, (min_x,max_y-start), (max_x,max_y-start), 255, 2)
    cv.namedWindow('test',cv.WINDOW_FREERATIO) 
    cv.imshow('test',img)
    windowShape = (int(img.shape[1]*0.4),int(img.shape[0]*0.4)) 
    cv.resizeWindow('test',windowShape)

    # Plot containers
    plt.show()
    '''
    plt.clf()
    return stop-start



for folder in PATHES_LIST:   # Iterate over folders   
    data_path = r'D:\Konrad\TCM_scan\dash_skany' + folder
    create_missing_catalogs(data_path)
    files = list(os.listdir(data_path + r'/images'))
    print(" ")
    print("Processing:",data_path)
    print("Number of images:", len(files))

    # Prepare containers for each row 
    containers = [] 
    rows_names = []
    rows_blunt_values = []
    max_dim = max_dim_in_rows( data_path + r'/otsu_tooth') # Find max x and y for particular row 
    for container in max_dim:   # Create containers for cumulated images
        name, x, y = container
        blank_image = np.zeros((y,x), np.uint8)
        containers.append(blank_image)
        rows_names.append(name)
        rows_blunt_values.append(0)
    
    # Read previously created dataframe
    df = pd.read_csv(BROACH_CSV + folder + '.csv')  
    print(df[:3]) # Show few data in console

    # Combine teeth in each row
    for image_name in files: 
        print(image_name)
        base_name = image_name[:image_name.rfind('.')] # Get data from dataframe
        split_name = base_name.split('_')
        row = int(split_name[1])
        data = df[df['img_name'] == image_name]
        minx = data['minx']
        miny  = data['miny']
        maxx = data['maxx']
        maxy = data['maxy']
        stepienie = cv.imread(data_path + r'/stepienie_analyze/' + image_name,cv.IMREAD_GRAYSCALE)
        if stepienie is not None:
            for i,name in enumerate(rows_names):
                if int(row) == name:
                    #stepienie = fill_pixels_in_blunt(stepienie) # Fill holes on the bottom of the tooth
                    #stepienie = normalize_in_x(stepienie) # Add weights 
                    
                    # Draw for debuging
                    #cv.namedWindow('test2',cv.WINDOW_FREERATIO) 
                    #cv.imshow('test2', stepienie)
                    #cv.waitKey(0)

                    cv.normalize(stepienie, stepienie, 0, 8, norm_type = cv.NORM_MINMAX) # Normalize pixels on the image 
                    containers[i][int(miny):stepienie.shape[0]+int(miny), int(minx):stepienie.shape[1]+int(minx)] += stepienie # Combine images 
                    #containers[i][0:stepienie.shape[0], int(minx):stepienie.shape[1]+int(minx)] += stepienie # Combine images          
                    #cv.rectangle(containers[1],(int(minx),int(miny)),(int(maxx),int(maxy)),255,1) # Draw external border of each tooth - debuging only
                      
    # Calculate blut value for each row and store it 
    for i,name in enumerate(rows_names):
        rows_blunt_values[i] = draw_plot(containers[i],name)

    tooth_blunt_values = []
    for i, j in df.iterrows():
        tooth_name = j['img_name']
        base_name = tooth_name[:tooth_name.rfind('.')]
        split_name = base_name.split('_')
        row = int(split_name[1])
        print(row,rows_names.index(row))
        print(tooth_name,rows_blunt_values[rows_names.index(row)])
        tooth_blunt_values.append(rows_blunt_values[rows_names.index(row)])
   
    
    df['stepienie_w_rzedach'] = tooth_blunt_values
    CSV_NAME = data_path.split('.')[0]
    CSV_NAME = CSV_NAME[CSV_NAME.rfind('\\') + 1:] + '.csv'
    df.to_csv (BROACH_CSV + '\\' + CSV_NAME, index = False, header=True)
