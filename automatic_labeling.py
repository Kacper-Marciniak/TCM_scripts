'''
Program prepares labelme .json annotation files based on predictions 
of the detectron2 model. It requires extracted teeth images linked by 
"pth" directory. Overlapping instances of the same class are merged. 
Output is saved as an .json file (1 image = 1 .json with the same name). 
User can manually controll and edit generated annotations in labelme.
After controll it files from "pth" dir can be put into: 
'\traning_segmentation\data\anot' and utilized for datasets creation.
'''

import prepare_models
import os
import cv2 as cv
import numpy as np
from matplotlib.image import imread
import scipy.misc
from PIL import Image  
import glob
from matplotlib.image import imread

# Path to the folder with extracted teeth images utilized for automatic annotations
pth = r'D:\Konrad\TCM_scan\traning_segmentation\test3_alicona\\'

models = prepare_models.Models()
segmentation_predictor = models.preapre_segmentation_model()

# Utilized to create labelme json file
labelme_start = '''{
"version": "4.5.10",
"flags": {},
"shapes": [
'''
def labelme_end(filename,h,w):
    labelme_end = '''],
"imagePath": "{}",
"imageData": null,
"imageHeight": {},
"imageWidth": {}
'''.format(filename,h,w)
    return labelme_end
def add_points(label,contour):
    '''
    Approximate single the contour passed to the function.
    Use eps approximation parameter to adjust number of points 
    higher eps ==> less points
    '''
    eps = 0.005 
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, eps * peri, True)
    cont = np.vstack(approx).squeeze().tolist()
    content = '''
"label": "{}",
"points": {},
"group_id": null,
"shape_type": "polygon",
"flags": {}
'''.format(label,cont,'{}')
    return str(content)

failures_dictionary = {
    0:"wykruszenie",
    1:"narost",
    2:"stepienie",
    3:"zatarcie"   
}


for imageName in glob.glob(pth + '*png'): # Iterate over all images in folder
    print(imageName)
    im = cv.imread(imageName)
    outputs = segmentation_predictor(im) # Make prediction 
    name = imageName.split('\\')[-1]
    img = Image.fromarray(im)
    base_name = name.split('.')[0]
    # Get mask and label from the prediction
    pred_masks = outputs["instances"].to("cpu").pred_masks.numpy()
    pred_classes = outputs["instances"].to("cpu").pred_classes.numpy()
    num_instances = pred_masks.shape[0] 
    pred_masks = np.moveaxis(pred_masks, 0, -1)
    output = np.zeros_like(im)
    labelme_json = labelme_start # Create begening of the labelme json file
    
    # Contains merged bitmaps for particular failures classes
    outputs = {
        "wykruszenie": np.zeros_like(im),
        "narost": np.zeros_like(im),
        "stepienie": np.zeros_like(im),
        "zatarcie": np.zeros_like(im)
    }

    # Contains temporary data used during merging
    pred_masks_instance = {   
        "wykruszenie": [],
        "narost": [],
        "stepienie": [],
        "zatarcie": []
    }

    # Iterate over predicted defects, search for duplicated instances of the same class and merge them into single bitmap.
    for i in range(num_instances): 
        for class_id in failures_dictionary: # Iterate over all instances classes
            if(pred_classes[i] == class_id): 
                failure_class = failures_dictionary[class_id] # Get name of the current class
                pred_masks_instance[failure_class].append(pred_masks[:, :, i:(i+1)]) 
                outputs[failure_class] = np.where(pred_masks_instance[failure_class][-1] == True, 255, outputs[failure_class]) 

    # Iterate over failures classes and convert generated bitmaps, extract contours, approximate it with the points, save it to the .json file       
    for class_name in failures_dictionary.values():
        im = Image.fromarray(outputs[class_name])
        im = np.array(im) 
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY )   
        contours = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours[0]) != 0:
            for c in contours[0]:
                # Add contour to the labelme json file
                labelme_json += '{' # Open container for new instance    
                labelme_json += add_points(class_name,c)
                labelme_json += '},\n'

    # Prepare end of the labelme json file
    labelme_json = labelme_json[:-2]
    labelme_json += labelme_end(base_name + '.png',img.size[1],img.size[0]) # Add some additional information
    labelme_json += '}' # Clossing requires by .json format
    out_txt_name = pth + base_name  + '.json'
    # Save labelme json file
    file = open(out_txt_name, 'w')
    file.write(str(labelme_json))
    file.close() 

    