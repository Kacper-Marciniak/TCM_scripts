import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pandas as pd
from PIL import Image, ImageOps 
import prepare_models
# r'D:\Konrad\TCM_scan\dash_skany\C_short',
PATHES_LIST =  [
                r'D:\Konrad\TCM_scan\dash_skany\alicone_A',
                r'D:\Konrad\TCM_scan\dash_skany\alicone_B',
                r'D:\Konrad\TCM_scan\dash_skany\A_new',
                r'D:\Konrad\TCM_scan\dash_skany\A_old',
                r'D:\Konrad\TCM_scan\dash_skany\B_old',  
                r'D:\Konrad\TCM_scan\dash_skany\brudny',
                r'D:\Konrad\TCM_scan\dash_skany\C_old',
                r'D:\Konrad\TCM_scan\dash_skany\D_new',
                r'D:\Konrad\TCM_scan\dash_skany\schodek',
                ] 

DASH_PATH = r'D:\Konrad\TCM_scan\dash' # Path to the folder with .csv files
BROACH_DIR = r'D:\Konrad\TCM_scan\dash_skany'  # Path to the corresponding folders with images 

def create_missing_catalogs(DATA_FOLDER_PATH):
    # Creates folders used for data saving and processing if sth is missing 
    if os.path.exists (DATA_FOLDER_PATH + r'\images') == False:     os.mkdir(DATA_FOLDER_PATH+ r'\images')
    if os.path.exists (DATA_FOLDER_PATH + r'\otsu_tooth') == False:     os.mkdir(DATA_FOLDER_PATH + r'\otsu_tooth')
    if os.path.exists (DATA_FOLDER_PATH + r'\segmentation') == False:       os.mkdir(DATA_FOLDER_PATH + r'\segmentation')
    if os.path.exists (DATA_FOLDER_PATH + r'\stepienie_analyze') == False:       os.mkdir(DATA_FOLDER_PATH + r'\stepienie_analyze')
    if os.path.exists (DATA_FOLDER_PATH + r'\plots') == False:       os.mkdir(DATA_FOLDER_PATH + r'\plots')  
def decode_segmentation(im, imageName):
  
    outputs = segmentation_predictor(im) # Inference 
    base_name = imageName.split('.')[0]
    # Get data from inference
    pred_masks = outputs["instances"].to("cpu").pred_masks.numpy()
    scores = outputs["instances"].to("cpu").scores.numpy()
    pred_classes = outputs["instances"].to("cpu").pred_classes.numpy()
    num_instances = pred_masks.shape[0] 
    pred_masks = np.moveaxis(pred_masks, 0, -1)

    # Save all instances as single masks in the 'segmentation' directory
    pred_masks_instance = []
    output = np.zeros_like(im)
    for i in range(num_instances):  # Iterate over instances and save detectron binary masks as images
        pred_masks_instance.append(pred_masks[:, :, i:(i+1)])
        output = np.where(pred_masks_instance[0] == True, 255, output)
        im = Image.fromarray(output)
        output = np.zeros_like(im)
        pred_masks_instance = []
        out_png_name = data_path + r'\segmentation' + '\\' + base_name + '-' + str(i) + '.png'
        im.save(out_png_name)

    # Combine instances 'stępienie' and save it for the further rows analyzys in 'stepienie_analyze' directory
    pred_masks_instance_stepienie = []
    output_stepienie = np.zeros_like(im)
    j = 0
    for i in range(num_instances): # Combine each 'stepienie' binary mask
        if(pred_classes[i] == 2):
            pred_masks_instance_stepienie.append(pred_masks[:, :, i:(i+1)])
            output_stepienie = np.where(pred_masks_instance_stepienie[j] == True, 255, output_stepienie)
            j+=1
    blunt_value = 0
    if(2 in pred_classes): # If there was at least 1 'stepienie' instance save results
        im = Image.fromarray(output_stepienie)
        out_png_name = data_path + r'\stepienie_analyze' + '\\' + base_name  + '.png'
        im = ImageOps.grayscale(im)
        im.save(out_png_name)

        img = cv.imread(out_png_name,cv.IMREAD_GRAYSCALE)

        blunt_value = analyze_blunt(img)
    return num_instances, pred_classes, scores, blunt_value
def analyze_blunt(img):
    # Find global bounding box containing all 'stepienie' instances

    contours, hierarchy = cv.findContours(img,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0 
    height, width = img.shape
    min_x, min_y = width, height
    max_x = max_y = 0
    for cnt in contours:
        idx += 1
        x,y,w,h = cv.boundingRect(cnt)
        min_x, max_x = min(x, min_x), max(x+w, max_x) # Find global min and max for bounding box 
        min_y, max_y = min(y, min_y), max(y+h, max_y) 
    return max_y - min_y
def add_instances_categories(df):
    # Add binary categories of the particular instances presence on the teeth
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
####################################################
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
    contours, hierarchy = cv.findContours(img,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)[-2:]
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

    # Find global bounding box containing all 'blunt' instances
    contours, hierarchy = cv.findContours(img,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)[-2:]
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
    cv.rectangle(img, (min_x-8, min_y-8), (max_x+8, max_y+8), 255, 4) # Draw bounding box
    cv.line(img, (min_x,max_y-stop), (max_x,max_y-stop), 255, 2)
    cv.line(img, (min_x,max_y-start), (max_x,max_y-start), 255, 2)
    cv.namedWindow('test',cv.WINDOW_FREERATIO) 
    cv.imshow('test',img)
    windowShape = (int(img.shape[1]*0.4),int(img.shape[0]*0.4)) 
    cv.resizeWindow('test',windowShape)

    # Plot containers
    plt.show()
    '''
    plt.clf()
    return stop-start

          

models = prepare_models.Models()
extraction_predictor = models.preapre_extraction_model()
segmentation_predictor = models.preapre_segmentation_model()

for data_path in PATHES_LIST:   # Iterate over folders  
    create_missing_catalogs(data_path)
    files = list(os.listdir(data_path + r'/images'))
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
        
        im = cv.imread(data_path + r'/images/' + image_name) # Read image
        print(image_name) 
        img_name.append(str(image_name))
        l_id.append(int(split_name[0]))
        w_id.append(int(split_name[1]))
        try: #Extraction
            outputs = extraction_predictor(im)
            minx, miny, maxx, maxy = list(list(outputs["instances"].to("cpu").pred_boxes)[0].numpy())
            roi = im.copy()[int(miny)-50:int(maxy)+50, int(minx)-100:int(maxx)+100]     
            cv.imwrite(data_path + r'\otsu_tooth' + '\\' + image_name , roi)
            min_x.append(int(minx))
            min_y.append(int(miny))
            max_x.append(int(maxx)) 
            max_y.append(int(maxy))              
            l.append(maxy - miny)
            w.append(maxx - minx)
            c_l.append((maxy + miny)/2)
            c_w.append((maxx + minx)/2)  
            try: #Segmentation
                num_instances, pred_class, score, blunt_value = decode_segmentation(roi, image_name)
                inst_num.append(str(num_instances))
                scores.append(str(score))
                inst_id.append(str(pred_class))
                blunt_values.append(blunt_value/603)
            except:
                inst_num.append(0)
                scores.append([0])
                inst_id.append([])
                blunt_values.append(-1)
        except:
            print("Extraction error in tooth:",image_name)   
            min_x.append(0)
            min_y.append(0)
            max_x.append(0) 
            max_y.append(0)              
            l.append(0)
            w.append(0)
            c_l.append(0)
            c_w.append(0)  
            inst_num.append(0)
            scores.append([0])
            inst_id.append([])
            blunt_values.append(-1)
                   
            # Preparing data for dataframe    
    data = {'img_name':img_name, 'minx':min_x,        'maxx':max_x,    'miny':min_y ,
            'maxy':max_y,        'l_id':l_id,         'w_id':w_id,     'l':l, 
            'w':w,               'c_l':c_l,           'c_w':c_w,       'inst_num':inst_num,     
            'scores':scores,     'inst_id':inst_id,   'wielkosc_stepienia':blunt_values}
    CSV_NAME = data_path.split('.')[0]
    CSV_NAME = CSV_NAME[CSV_NAME.rfind('\\') + 1:] + '.csv'
    df = pd.DataFrame(data, columns = ['img_name','minx','maxx','miny','maxy','l_id', 'w_id','l', 'w', 'c_l', 'c_w', 'inst_num','scores','inst_id','wielkosc_stepienia']) 
    print(df[:3]) # Show few data in console
    df = add_instances_categories(df) 
    print(df[:3]) # Show few data in console
    df.to_csv (DASH_PATH + '\\' + CSV_NAME, index = False, header=True)

    # Prepare containers for each row 
    containers, rows_names, rows_blunt_values = [], [], []
    max_dim = max_dim_in_rows( data_path + r'/otsu_tooth') # Find max x and y for particular row 
    for container in max_dim:   # Create containers for cumulated images
        name, x, y = container
        blank_image = np.zeros((y,x), np.uint8)
        containers.append(blank_image)
        rows_names.append(name)
        rows_blunt_values.append(0)
    
    # Read previously created dataframe
    df = pd.read_csv(DASH_PATH + '\\' + CSV_NAME)  
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
                if int(row) == name: # Numerous rows analyze options
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

    # Calculate blut value for each tooth and store it 
    tooth_blunt_values = []
    for i, j in df.iterrows():
        tooth_name = j['img_name']
        base_name = tooth_name[:tooth_name.rfind('.')]
        split_name = base_name.split('_')
        row = int(split_name[1])
        print(row,rows_names.index(row))
        print(tooth_name,rows_blunt_values[rows_names.index(row)])
        tooth_blunt_values.append(rows_blunt_values[rows_names.index(row)])
   
    # Add data to the .csv
    df['stepienie_w_rzedach'] = tooth_blunt_values
    CSV_NAME = data_path.split('.')[0]
    CSV_NAME = CSV_NAME[CSV_NAME.rfind('\\') + 1:] + '.csv'
    df.to_csv (DASH_PATH + '\\' + CSV_NAME, index = False, header=True)
    
