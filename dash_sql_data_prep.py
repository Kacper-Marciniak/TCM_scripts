import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from matplotlib.image import imread
import scipy.misc
from PIL import Image, ImageOps 
import prepare_models
import sql_connection
import shutil

# User define variables
SCAN_NAME = '22-03-23-13-12'

# Program variables
INPUT_PATH = r'D:\Konrad\TCM_scan\dash_skany\A_new\images'    # Path to the folder with the images directly from scaner
OUTPUT_IMG_PATH  = r'D:\Konrad\TCM_scan\dash_sql/'   # Path to the folder with processed images_F
OUTPUT_IMG_PATH = OUTPUT_IMG_PATH + SCAN_NAME

models = prepare_models.Models()
sql = sql_connection.SQLConnection(debug=False)
sql.create_scan(SCAN_NAME,OUTPUT_IMG_PATH)
extraction_predictor = models.preapre_extraction_model()
segmentation_predictor = models.preapre_segmentation_model()
if os.path.exists (OUTPUT_IMG_PATH) == False:os.mkdir(OUTPUT_IMG_PATH)
models.create_missing_catalogs(OUTPUT_IMG_PATH) # Create subdirectories for the SCAN_NAME directory

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
        out_png_name = OUTPUT_IMG_PATH + r'\segmentation' + '\\' + base_name + '-' + str(i) + '.png'
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
        out_png_name = OUTPUT_IMG_PATH + r'\stepienie_analyze' + '\\' + base_name  + '.png'
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
def add_binary_categories(inst_ids):
    # Add binary categories of the particular instances presence on the teeth
    inst_ids = inst_ids[inst_ids.rfind('[') + 1:]
    inst_ids = inst_ids[:inst_ids.rfind(']')]
    inst_ids = np.array(inst_ids.split(' '))
    
    # if '2' in inst_ids: stepienie = 1 # No need to calculate binary presence of the stepeinie class (there is full information)
    # else: stepienie = 1
    if '1' in inst_ids: narost = 1
    else: narost = 0 
    if '3' in inst_ids: zatarcie = 1
    else: zatarcie = 0  
    if '4'in inst_ids: wykruszenie = 1
    else: wykruszenie = 0

    return narost, zatarcie, wykruszenie
def tooth_inference(image_name):
    try:
        im = cv.imread(INPUT_PATH + '//' + image_name) # Read image
        cv.imwrite(OUTPUT_IMG_PATH + '/images/' + image_name, im)
        outputs = extraction_predictor(im)
        minx, miny, maxx, maxy = list(list(outputs["instances"].to("cpu").pred_boxes)[0].numpy())
    except:
        print("There is problem with image:",INPUT_PATH + '//' + image_name)
        try:
            source = INPUT_PATH + '//' + image_name
            destination = OUTPUT_IMG_PATH + '/images/' + image_name 
            shutil.copyfile(source, destination)
        except:
            print("Can't copy corrupted file")


    try: # Extraction
        roi = im.copy()[int(miny)-50:int(maxy)+50, int(minx)-100:int(maxx)+100] 
        cv.imwrite(OUTPUT_IMG_PATH + '/otsu_tooth/' + image_name , roi) 

        length = (maxy - miny)/603
        width = (maxx - minx)/603
        centre_lenght = ((maxy + miny)/2)/603
        centre_width = ((maxx + minx)/2)/603 

        length, width, centre_lenght, centre_width = round(length,5), round(width,5), round(centre_lenght,5), round(centre_width,5) 
        try: # Segmentation
            num_instances, pred_class, score, stepienie = decode_segmentation(roi, image_name)
            num_instances = str(num_instances)
            score = [ round(elem, 4) for elem in score ]
            score = str(score)
            pred_class = str(pred_class)
            stepienie = stepienie/603
            stepienie = round(stepienie,5)
            narost, zatarcie, wykruszenie = add_binary_categories(pred_class)
        except:
            print("Segmentation error in tooth:", image_name)  
            num_instances = None
            score = None
            pred_class = None
            stepienie = None
            narost, zatarcie, wykruszenie = None,None,None
    except:
        print("Extraction error in tooth:", image_name)  
        length = None
        width = None
        centre_lenght = None
        centre_width = None 
        num_instances = None
        score = None
        pred_class = None
        stepienie = None
        narost, zatarcie, wykruszenie = None,None,None

    dict = {'length':length, 'width':width, 'centre_lenght':centre_lenght, 'centre_width':centre_width, 
            'num_instances':num_instances, 'score':score, 'pred_class':pred_class, 'stepienie': stepienie, 
            'narost':narost, 'zatarcie':zatarcie, 'wykruszenie':wykruszenie, 'image_name':image_name }
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
        maxy = (2*centre_lenght+length)/2*603
        miny = (2*centre_lenght-length)/2*603
        maxx = (2*centre_width+width)/2*603
        minx = (2*centre_width-width)/2*603

        # Combine all images in row 
        containers = np.zeros((2748,3840), np.uint8)
        for i,image_name in enumerate(image_names): 
            
            img_path = OUTPUT_IMG_PATH + r'/stepienie_analyze/' + image_name
            stepienie = cv.imread(img_path,cv.IMREAD_GRAYSCALE)
            cv.normalize(stepienie, stepienie, 0, 8, norm_type = cv.NORM_MINMAX) # Normalize pixels on the image 
            containers[int(miny[i]):stepienie.shape[0]+int(miny[i]), int(minx[i]):stepienie.shape[1]+int(minx[i])] += stepienie # Combine images 

        # Calculate combined 'stepienie' size
        row_stepienie = draw_plot(containers,row_number) 
        print('row_number: {} stepienie: {:0.3f} mm'.format(row_number,row_stepienie))
        sql.add_row_param(SCAN_NAME, row_stepienie, row_number)
def draw_plot(img,name):

    # Find global bounding box containing all 'stepienie' instances
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
    #print("start: {} px, stop: {}, dif {:0.3f} mm".format(start,stop,(stop-start)/603))
    plot_name = OUTPUT_IMG_PATH + r'/plots/' + str(name) + '.jpg' 
    plt.plot(bins)
    plt.axhline(y=int(max_v*0.1), color='r', linestyle='-') # 10% line
    plt.axvline(x=start, color='r', linestyle='--') # 10% line
    plt.axvline(x=stop, color='r', linestyle='--') # 10% line
    plt.title("Row stępienie = {:0.3f}mm".format((stop-start)/603))
    plt.axis("off")
    plt.savefig(plot_name,dpi=300)
    plt.clf()
    
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
    plt.clf()
    '''
    return (stop-start)/603

    '''
    print(' ')
    print('id:',sql.get_row_param(SCAN_NAME,'id','row_number=1'))
    print(' ')
    print(sql.get_tooth_param(SCAN_NAME,'tooth_number','1'))
    '''


files = list(os.listdir(INPUT_PATH))
for image_name in files: # Process all available images
    base_name = image_name[:image_name.rfind('.')]
    split_name = base_name.split('_')
    row = int(split_name[1])
    section = int(split_name[0])
        
    # Find single tooth parameters
    tooth_inference(image_name)

# Find row parameters
row_inference()
    
