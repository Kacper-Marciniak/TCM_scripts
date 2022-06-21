import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import prepare_models
import sql_connection
import shutil
import datetime

from tkinter_dialog_custom import askdirectory
from PARAMETERS import PATH_DASH_SQL_DIR

# User define variables
SCAN_NAME = str(datetime.datetime.now().strftime("%m%d%Y-%H%M%S"))    # Current date as scan name

# Program variables
INPUT_PATH = askdirectory(title="Select folder with images from scanner").replace('\\\\','\\').replace('\\','/')    # Path to the folder with the images directly from scaner
OUTPUT_IMG_PATH  = askdirectory(title="Select DASH-SQL server to write to", initialdir = PATH_DASH_SQL_DIR)  # Path to the folder with processed images_F
OUTPUT_IMG_PATH = os.path.join(OUTPUT_IMG_PATH,SCAN_NAME).replace('\\\\','\\').replace('\\','/')

# ROI extraction offsets
min_x_off, min_y_off, max_x_off, max_y_off = 100, 50, 100, 50 # offseting the ROI bounding box

models = prepare_models.Models()
sql = sql_connection.SQLConnection(debug=False)
sql.create_scan(SCAN_NAME,OUTPUT_IMG_PATH)
extraction_predictor = models.preapre_extraction_model()
segmentation_predictor = models.preapre_segmentation_model()
if os.path.exists (OUTPUT_IMG_PATH) == False: os.mkdir(OUTPUT_IMG_PATH)
models.create_missing_catalogs(OUTPUT_IMG_PATH) # Create subdirectories for the SCAN_NAME directory

def decode_segmentation(im, imageName):  
    outputs = segmentation_predictor(im) # Inference 
    base_name = imageName.split('.')[0]
    # Get data from inference
    outputs_instances = outputs["instances"].to("cpu")
    pred_masks = outputs_instances.pred_masks.numpy()
    scores = outputs_instances.scores.numpy()
    pred_classes = outputs_instances.pred_classes.numpy()
    num_instances = pred_masks.shape[0] 
    pred_masks = np.moveaxis(pred_masks, 0, -1)

    # Save all instances as single masks in the 'segmentation' directory
    output = np.zeros_like(im)
    for i in range(num_instances):  # Iterate over instances and save detectron binary masks as images
        output_mask = np.where(pred_masks[:, :, i:(i+1)] == True, 255, output)
        out_png_name = OUTPUT_IMG_PATH+"/segmentation/"+base_name+"_"+str(i)+".png"
        if not (output_mask.size == 0):
            cv.imwrite(out_png_name, output_mask)

    # Combine instances 'stępienie' and save it for the further rows analyzys in 'stepienie_analyze' directory
    pred_masks_instance_stepienie = []
    output_stepienie = np.zeros_like(im)
    j = 0
    blunt = False # check if there is at least 1 valid blunt
    for i in range(num_instances): # Combine each 'stepienie' binary mask
        if(pred_classes[i] == 2):
            pred_masks_instance_stepienie.append(pred_masks[:, :, i:(i+1)])
            out_tpl = np.nonzero(pred_masks_instance_stepienie[j])        
            if max(out_tpl[1])/pred_masks_instance_stepienie[j].shape[1] > 0.8: # check if the blunt is at the bottom of the image if not - skip (it can be improved)
                output_stepienie = np.where(pred_masks_instance_stepienie[j] == True, 255, output_stepienie)
                blunt = True
            j+=1
    blunt_value = 0

    if(blunt): # If there was at least 1 'stepienie' instance save results
        out_png_name = OUTPUT_IMG_PATH+"/stepienie_analyze/"+ base_name+".png"
        output_stepienie = cv.cvtColor(output_stepienie, cv.COLOR_BGR2GRAY)
        cv.imwrite(out_png_name, output_stepienie)
        blunt_value = analyze_blunt(output_stepienie)
    return num_instances, pred_classes, scores, blunt_value

def analyze_blunt(img):
    # Find global bounding box containing all 'stepienie' instances

    contours, _ = cv.findContours(img,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0 
    min_x, min_y = img.shape[1], img.shape[0]
    max_x = max_y = 0
    for cnt in contours:
        idx += 1
        x,y,w,h = cv.boundingRect(cnt)
        min_x, max_x = min(x, min_x), max(x+w, max_x) # Find global min and max for bounding box 
        min_y, max_y = min(y, min_y), max(y+h, max_y) 
    return max_y - min_y

def add_binary_categories(inst_ids):
    # Add binary categories of the particular instances presence on the teeth
    inst_ids = np.array(inst_ids[inst_ids.rfind('[') + 1: inst_ids.rfind(']')].split(' '))
    
    # if '2' in inst_ids: stepienie = 1 # No need to calculate binary presence of the stepeinie class (there is full information)
    # else: stepienie = 1
    narost, zatarcie, wykruszenie = 0, 0, 0
    if '1' in inst_ids: narost = 1
    if '3' in inst_ids: zatarcie = 1
    if '4'in inst_ids: wykruszenie = 1

    return narost, zatarcie, wykruszenie

def tooth_inference(image_name):
    try: # Extraction predictor
        im = cv.imread(os.path.join(INPUT_PATH,image_name)) # Read image
        im_to_save = compress_image(im)
        cv.imwrite(OUTPUT_IMG_PATH+'/images/'+image_name, im_to_save)
        outputs = extraction_predictor(im) # extraction
        min_x, min_y, max_x, max_y = list(list(outputs["instances"].to("cpu").pred_boxes)[0].numpy())
    except Exception as e:
        print(f"There is problem with image: {INPUT_PATH}/{image_name}\n\t\tException: {e}")
        min_x, min_y, max_x, max_y  = 0, 0, 0, 0
        try:
            source = os.path.join(INPUT_PATH, image_name)
            destination = OUTPUT_IMG_PATH+"/images/"+image_name
            shutil.copyfile(source, destination)
        except:
            print("Can't copy corrupted file")


    try: # Segmentation
        if not (min_x == min_y == max_x == max_y): 
            roi = im.copy()[int(min_y)-min_y_off:int(max_y)+max_y_off, int(min_x)-min_x_off:int(max_x)+min_x_off]
        else: roi = im.copy()
        cv.imwrite(OUTPUT_IMG_PATH+"/otsu_tooth/"+image_name, roi) 
        length = (max_y - min_y)/603
        width = (max_x - min_x)/603
        centre_lenght = ((max_y + min_y)/2)/603
        centre_width = ((max_x + min_x)/2)/603 

        length, width, centre_lenght, centre_width = round(length,5), round(width,5), round(centre_lenght,5), round(centre_width,5) 
        try: # Segmentation
            num_instances, pred_class, score, stepienie = decode_segmentation(roi, image_name)
            num_instances = str(num_instances)
            score = str([round(elem, 4) for elem in score])
            pred_class = str(pred_class)
            stepienie = stepienie/603
            stepienie = round(stepienie,5)
            narost, zatarcie, wykruszenie = add_binary_categories(pred_class)
        except Exception as e_segm:
            print(f"Segmentation error in tooth: {image_name},\n\t\tException: {e_segm}")  
            num_instances = None
            score = None
            pred_class = None
            stepienie = None
            narost, zatarcie, wykruszenie = None,None,None
    except Exception as e_ext:
        print(f"Extraction error in tooth: {image_name},\n\t\tException: {e_ext}")
        length = None
        width = None
        centre_lenght = None
        centre_width = None 
        num_instances = None
        score = None
        pred_class = None
        stepienie = None
        narost, zatarcie, wykruszenie = None,None,None

    dict = {
        'length':length, 
        'width':width,
        'centre_lenght':centre_lenght,
        'centre_width':centre_width, 
        'num_instances':num_instances,
        'score':score,
        'pred_class':pred_class,
        'stepienie': stepienie, 
        'narost':narost,
        'zatarcie':zatarcie,
        'wykruszenie':wykruszenie,
        'image_name':image_name 
    }
    sql.add_tooth(image_name,dict)

def convert_sql_output(sql_data):
    sql_data = np.array(sql_data)
    sql_data = np.reshape(sql_data,(-1))
    return sql_data

def row_inference():
  
    # Get all rows numbers from sql
    available_rows = convert_sql_output(sql.get_row_param(SCAN_NAME,'row_number'))

    # Iterate over all rows
    for row_number in available_rows: 

        # Find image parameters for teeth with non zero stepienie value
        image_names = convert_sql_output(sql.get_tooth_param(scan_name = SCAN_NAME, param_name = 'image_name', row_number = str(row_number),conditions = 'stepienie>0'))
        centre_lenght = convert_sql_output(sql.get_tooth_param(scan_name = SCAN_NAME, param_name = 'centre_lenght', row_number = str(row_number),conditions = 'stepienie>0'))
        length = convert_sql_output(sql.get_tooth_param(scan_name = SCAN_NAME, param_name = 'length', row_number = str(row_number),conditions = 'stepienie>0'))
        centre_width = convert_sql_output(sql.get_tooth_param(scan_name = SCAN_NAME, param_name = 'centre_width', row_number = str(row_number),conditions = 'stepienie>0'))
        width = convert_sql_output(sql.get_tooth_param(scan_name = SCAN_NAME, param_name = 'width', row_number = str(row_number),conditions = 'stepienie>0'))

        # Calculate values needed for images overlaying
        #max_y = (2*centre_lenght+length)/2*603
        min_y = (2*centre_lenght-length)/2*603
        #max_x = (2*centre_width+width)/2*603
        min_x = (2*centre_width-width)/2*603

        # Combine all images in row 
        containers = np.zeros((2748,3840), np.uint8)
        for i,image_name in enumerate(image_names): 
            
            img_path = f"{OUTPUT_IMG_PATH}/stepienie_analyze/{image_name}"
            stepienie = cv.imread(img_path,cv.IMREAD_GRAYSCALE)
            cv.normalize(stepienie, stepienie, 0, 8, norm_type = cv.NORM_MINMAX)
            # Normalize pixels on the image 
            # Previous extracted img borders expanding cause some problems with big teeth with lay very close to the border
            # If those tooth would lay to close or to far move it in one or other way
            try:
                containers[int(min_y[i]):stepienie.shape[0]+int(min_y[i]), int(min_x[i]):stepienie.shape[1]+int(min_x[i])] += stepienie # Combine images 
            except:
                try:
                    containers[int(min_y[i]):stepienie.shape[0]+int(min_y[i]), int(min_x[i])-100:stepienie.shape[1]+int(min_x[i])-100] += stepienie # Combine images 
                except:
                    containers[int(min_y[i]):stepienie.shape[0]+int(min_y[i]), int(min_x[i])+100:stepienie.shape[1]+int(min_x[i])+100] += stepienie # Combine images 
        # Calculate combined 'stepienie' size
        row_stepienie = draw_plot(containers,row_number) 
        print(f'row_number: {row_number} stepienie: {row_stepienie:0.3f} mm')
        sql.add_row_param(SCAN_NAME, row_stepienie, row_number)

def draw_plot(img,name):

    # Find global bounding box containing all 'stepienie' instances
    contours, _ = cv.findContours(img,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0 
    min_x, min_y = img.shape[1], img.shape[0]
    max_x = max_y = 0
    for cnt in contours:
        idx += 1
        x,y,w,h = cv.boundingRect(cnt)
        min_x, max_x = min(x, min_x), max(x+w, max_x) # Find global min and max 
        min_y, max_y = min(y, min_y), max(y+h, max_y) # for bounding box  
   
    if idx == 0: return 0 # If no contours --> stop 

    # Define boxes for cumulative chart 
    bins = [0]*(max_y-min_y)

    roi = img[min_y:max_y,min_x:max_x] # Internal area of the bounding box    
    max_v = 0 # Max piksels in 1 container
    for py in range(0,max_y-min_y):
        for px in range (0,max_x-min_x):
            bins[len(bins)-py-1] += roi[py][px]
        max_v = max(max_v, bins[len(bins)-py-1])
        
    stop = max([i for i,b in enumerate(bins) if b > max_v*0.1]) # Find max bin index with values above 10% 
    start = min([i for i,b in enumerate(bins) if b > max_v*0.1]) # Find max bin index with values above 10% 
    
    plot_name = f'{OUTPUT_IMG_PATH}/plots/{str(name)}.jpg' 
    plt.plot(bins)
    plt.axhline(y=int(max_v*0.1), color='r', linestyle='-') # 10% line
    plt.axvline(x=start, color='r', linestyle='--') # 10% line
    plt.axvline(x=stop, color='r', linestyle='--') # 10% line
    plt.title(f"Row stępienie = {((stop-start)/603):0.3f} mm")
    plt.axis("off")
    plt.savefig(plot_name,dpi=300)
    plt.clf()
    

    return (stop-start)/603

## COMPRESSION
JPEG_COMPRESSION_QUALITY = 50
PNG_SAVING_COMPRESSION = 3 # 0-fast -> 9-good compression
RESIZE_SCALE = 4

def compress_image(img, path_tmp_file=""):
    if path_tmp_file == "":  path_tmp_file=  os.path.join(os.path.dirname(os.path.abspath(__file__)),"temp.jpg")
    w,h = img.shape[1],img.shape[0]
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img,[w//RESIZE_SCALE,h//RESIZE_SCALE], interpolation=cv.INTER_NEAREST)
    # save file with compression as jpeg
    cv.imwrite(
        path_tmp_file, 
        img, 
        [cv.IMWRITE_JPEG_QUALITY, JPEG_COMPRESSION_QUALITY, 
        cv.IMWRITE_JPEG_OPTIMIZE, 1,
        cv.IMWRITE_JPEG_PROGRESSIVE, 1], 
    )
    img = cv.imread(path_tmp_file, 0) # flag 0 - read as grayscale
    # remove temp jpeg file
    os.remove(path_tmp_file)
    img = cv.resize(img,[w,h], interpolation=cv.INTER_NEAREST)
    return img


list_files = list(os.listdir(INPUT_PATH))

for i,image_name in enumerate(list_files): # Process all available images
    base_name = image_name.split('.')[-2]
    split_name = base_name.split('_')
    row = int(split_name[-1])
    section = int(split_name[-2])
    
    print(f"{i+1}/{len(list_files)}")

    # Find single tooth parameters
    tooth_inference(image_name)

# Find row parameters
row_inference()