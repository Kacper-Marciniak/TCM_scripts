  
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
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

##--------------------------------------------------------------------------------------------------------------------------------------------------##
# Input parameters

PATHES_LIST =  [r'D:\Konrad\TCM_scan\MSA_WS\MSA_type1',
                r'D:\Konrad\TCM_scan\MSA_WS\MSA_rr\pwr_c_odtwarzalnosc_1_ws',
                r'D:\Konrad\TCM_scan\MSA_WS\MSA_rr\pwr_c_odtwarzalnosc_2_ws',
                r'D:\Konrad\TCM_scan\MSA_WS\MSA_rr\pwr_c_odtwarzalnosc_3_ws',
                r'D:\Konrad\TCM_scan\MSA_WS\MSA_rr\pwr_c_odtwarzalnosc_4_kc',
                r'D:\Konrad\TCM_scan\MSA_WS\MSA_rr\pwr_c_odtwarzalnosc_5_kc',
                r'D:\Konrad\TCM_scan\MSA_WS\MSA_rr\pwr_c_odtwarzalnosc_6_kc'] # Foldery do analizy, wpisz ile chesz - zrobi wszystkie jeden po drugim
'''
PATHES_LIST =  [r'D:\Konrad\TCM_scan\MSA_type1',
                r'D:\Konrad\TCM_scan\Skany_do_MSA\20211019_080642_C_2_ws',
                r'D:\Konrad\TCM_scan\Skany_do_MSA\20211019_101645_C_3_ws'] # Foldery do analizy, wpisz ile chesz - zrobi wszystkie jeden po drugim

UWAGA!
W folderze z danymi np: '20211018_142334_C_1_ws' musi być podfolder 'images' i dopiero w nim zdjecia
'''

NUM_SAMPLES = 100 # Ustaw dużą to zrobi wszystkie
DISPLAY = False # Wyświetlanie do debugowania 


'''
Wyniki będę w folderze: 
D:\Konrad\TCM_scan\broach_comparison
'''
##--------------------------------------------------------------------------------------------------------------------------------------------------##

#NUMBER_OF_ROWS = 1 #Ilość rzędów do skanowania
#NUMBER_OF_TEETHS_IN_COL = 84 #Ilość zębów w kolumnie przeciągacza
#NUM_SAMPLES = NUMBER_OF_TEETHS_IN_COL * NUMBER_OF_ROWS
class CocoTrainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"
    return COCOEvaluator(dataset_name, cfg, False, output_folder)

# Model preparation
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


# Iterate over folders and process each image, save results to csv 
for data_path in PATHES_LIST:
    print(" ")
    print("Processing:",data_path)
    min_x,min_y,max_x,max_y,x,y,xmm,ymm,n,name=[],[],[],[],[],[],[],[],[],[]
    files = list(os.listdir(data_path + r'\images'))
    print(files[:10])
    for i,image_name in enumerate(files):
        
        
        if i > NUM_SAMPLES: break
        img_path = data_path + r'\images' + '\\' + image_name 
        im = cv2.imread(img_path)
        outputs = predictor(im)
        #print(outputs)
    
        if DISPLAY:
            MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = ['tooth']
            v = Visualizer(im[:, :, ::-1],
                        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
                        scale = 0.3,
                        instance_mode = ColorMode.SEGMENTATION)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow(image_name,v.get_image()[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyWindow(image_name)
        
        try:
            minx,miny,maxx,maxy = list(list(outputs["instances"].to("cpu").pred_boxes)[0].numpy())
            #minx,miny,maxx,maxy = int(minx),int(miny),int(maxx),int(maxy)
            min_x.append(minx)
            max_x.append(maxx)
            min_y.append(miny)
            max_y.append(maxy)
            x.append(maxx-minx)
            y.append(maxy-miny)
            xmm.append((maxx-minx)/603)
            ymm.append((maxy-miny)/603)
            n.append((i % 96)+1)
            name.append(image_name)
        except:
            print("Error in tooth:",image_name)

    data = {'id':n,'name':name,'minx':min_x,'miny':min_y,'maxx':max_x,'maxy':max_y,'x':x,'y':y,'xmm':xmm,'ymm':ymm}
    df = pd.DataFrame(data, columns= ['id','name','minx','miny','maxx','maxy','x','y','xmm','ymm'])  

    CSV_name = r'D:\Konrad\TCM_scan\broach_comparison'+ '\\' + data_path[data_path.rfind('\\')+1:] + '.csv'
    f = open(CSV_name,'w')
    f.close()
    df.to_csv (CSV_name, index = False, header=True)
    print("Finished:",CSV_name)

print("Finished all")