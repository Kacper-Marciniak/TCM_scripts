r'''
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
from PIL import Image  

from tkinter_dialog_custom import askdirectory

from PARAMETERS import DICTIONARY_FAILURES
from PARAMETERS import LABELING_EPS_VALUE

# Path to the folder with extracted teeth images utilized for automatic annotations


path_to_data = askdirectory(title="Select path to input folder with images to annotate").replace('\\\\','\\').replace('\\','/')

if path_to_data == None: exit()

models = prepare_models.Models()
segmentation_predictor = models.preapre_segmentation_model()

failures_dictionary = DICTIONARY_FAILURES

# Utilized to create labelme json file
labelme_start = '''{
"version": "4.5.10",
"flags": {},
"shapes": [ 
'''
def labelme_end(filename,h,w):
    labelme_end = f'''],
"imagePath": "{filename}",
"imageData": null,
"imageHeight": {h},
"imageWidth": {w}
'''
    return labelme_end
def add_points(label,contour):
    '''
    Approximate a single contour passed to the function.
    Use eps approximation parameter to adjust number of points 
    higher eps ==> less points
    '''
    eps = LABELING_EPS_VALUE
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, eps * peri, True)
    cont = np.vstack(approx).squeeze().tolist()
    content = f'''
"label": "{label}",
"points": {cont},
"group_id": null,
"shape_type": "polygon",
"flags": {r'{}'}
'''
    return content

list_images = list() # store all image files
for filename in os.listdir(path_to_data): # Iterate over all images in folder
    if not ".png" in filename: continue
    list_images.append(filename)

for idx_file,imageName in enumerate(list_images): # Iterate over all images in folder
    print(f"Auto labeling image: {imageName} --- {idx_file+1}/{len(list_images)}")
    im = cv.imread(os.path.join(path_to_data,imageName))
    outputs = segmentation_predictor(im) # Make prediction 
    name = imageName.split('/')[-1]
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
        DICTIONARY_FAILURES[0]: np.zeros_like(im),
        DICTIONARY_FAILURES[1]: np.zeros_like(im),
        DICTIONARY_FAILURES[2]: np.zeros_like(im),
        DICTIONARY_FAILURES[3]: np.zeros_like(im)
    }
    
    # Contains temporary data used during merging
    pred_masks_instance = {   
        DICTIONARY_FAILURES[0]: [],
        DICTIONARY_FAILURES[1]: [],
        DICTIONARY_FAILURES[2]: [],
        DICTIONARY_FAILURES[3]: []
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
        im = cv.cvtColor(np.array(Image.fromarray(outputs[class_name])) , cv.COLOR_BGR2GRAY )   
        contours = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours[0]) != 0:
            for c in contours[0]:
                # Add contour to the labelme json file
                labelme_json += "{" + add_points(class_name,c) + "},\n"
                # Open container for new instance 

    # Prepare end of the labelme json file
    labelme_json = labelme_json[:-2]
    labelme_json += labelme_end(f"{base_name}.png",img.size[1],img.size[0]) # Add some additional information
    labelme_json += '}' # Clossing requires by .json format
    out_txt_name = os.path.join(path_to_data, f"{base_name}.json")
    # Save labelme json file
    file = open(out_txt_name, 'w')
    file.write(str(labelme_json))
    file.close() 
    print(f"{out_txt_name} generated!")

print("ALL DATA LABELED")
    