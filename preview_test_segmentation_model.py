import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import prepare_models

from tkinter_dialog_custom import askdirectory
from PARAMETERS import PATH_TRAINING_DATA_SEGMENTATION
from PARAMETERS import DICTIONARY_FAILURES
from PARAMETERS import LABELING_EPS_VALUE

import json

# Program variables
INPUT_PATH = askdirectory(title="Select folder with images", initialdir=PATH_TRAINING_DATA_SEGMENTATION).replace('\\\\','\\').replace('\\','/')    # Path to the folder with the images directly from scaner

COLOUR = dict()
COLOUR["stepienie"] = (255,0,0)
COLOUR["zatarcie"] = (0,255,0)
COLOUR["wykruszenie"] = (0,0,255)
COLOUR["narost"]= (255,255,0)

models = prepare_models.Models()

segmentation_predictor = models.preapre_segmentation_model()

def aprox_with_contour(bit_mask):
    contours,_ = cv.findContours(bit_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    list_points = list()
    for contour in contours:
        contour = contour.squeeze().round().astype(np.int32).tolist()
        list_points.append(contour)

    return list_points

def apply_masks_on_image(im,output_masks):
    im_preview = np.zeros_like(im)

    colours_classes = dict()
    for classname in DICTIONARY_FAILURES.values():
        colours_classes[classname] = np.zeros_like(im)

    colours_classes["stepienie"][:,:,0] = 255
    colours_classes["zatarcie"][:,:,1] = 255
    colours_classes["wykruszenie"][:,:,2] = 255
    colours_classes["narost"][:,:,0] , colours_classes["narost"][:,:,1] = 255, 255

    for failure in DICTIONARY_FAILURES.values():   
            mask = output_masks[failure]
            colour = colours_classes[failure]
            im_preview = cv.add(
                    cv.bitwise_and(im_preview, im_preview, mask=cv.bitwise_not(mask)),
                    cv.bitwise_and(colour,colour,mask=mask),
            )

    im = cv.addWeighted(
        im_preview, 1.0,
        im,0.75,
        0
    )

    return im

def decode_segmentation(im, imageName):  
    outputs = segmentation_predictor(im) # Inference 
    # Get data from inference
    outputs_instances = outputs["instances"].to("cpu")
    pred_masks = outputs_instances.pred_masks.numpy()
    pred_classes = outputs_instances.pred_classes.numpy()
    scores = outputs_instances.scores.numpy()
    print(f"image: {imageName}")
    for score, classid in zip(scores,pred_classes):
        classname = DICTIONARY_FAILURES[classid]
        print(f"\tDetected: {classname}\n\tScore: {score}")

    output_masks = dict()
    output_points = dict()
    for failure in DICTIONARY_FAILURES.values():
        output_masks[failure]=np.zeros((im.shape[0],im.shape[1],1), dtype=np.uint8)
        output_points[failure]=list()

    # Save all instances as single masks in the 'segmentation' directory
    for mask, class_id in zip(pred_masks, pred_classes):  # Iterate over instances and save detectron binary masks as images
        output_mask = np.where(np.expand_dims(mask, axis=2) == True, 255, 0).astype(np.uint8)
        class_name = DICTIONARY_FAILURES[class_id]
        output_masks[class_name] = cv.bitwise_or(output_masks[class_name],output_mask)
        output_points[class_name] += aprox_with_contour(output_mask)

    return output_masks, output_points

def test_segmentation(im, image_name):

    output_masks, output_points = decode_segmentation(im, image_name)
    im_preview = apply_masks_on_image(im, output_masks)
    
    return im_preview, output_points

def get_labeled_contours(im, image_name):
    json_name = image_name.replace("png","json")

    output_masks, output_points = decode_json(im, json_name)
    im_preview = apply_masks_on_image(im, output_masks)
    
    return im_preview, output_points

def decode_json(im, json_name):    
    output_masks = dict()
    output_points = dict()
    for failure in DICTIONARY_FAILURES.values():
        output_masks[failure]=np.zeros((im.shape[0],im.shape[1],1), dtype=np.uint8)
        output_points[failure] = list()
    
    with open(os.path.join(INPUT_PATH,json_name), 'r') as json_file:
        json_label = json.load(json_file)
        if len(json_label["shapes"]) == 0:
            return output_masks[classname]
        for shape_dict in json_label["shapes"]:
            classname = shape_dict["label"]
            points = np.array(shape_dict["points"]).round()
            output_points[classname].append(points)   
            
            cv.fillPoly(img=output_masks[classname], pts=np.int32([points]), color=255)
        
    return output_masks, output_points



list_files = list(os.listdir(INPUT_PATH))

for i,image_name in enumerate(list_files):
    if not ".png" in image_name:
        list_files.remove(image_name)

for i,image_name in enumerate(list_files): # Process all available images
    print(f"{i+1}/{len(list_files)}")

    im = cv.imread(os.path.join(INPUT_PATH,image_name)) # Read image

    # Find single tooth parameters
    im_predicted, predicted_cnt = test_segmentation(im, image_name)
    im_labeled, labeled_cnt = get_labeled_contours(im, image_name)
    im_predicted_preview = im.copy()
    im_labeled_preview = im.copy()
    im_both_preview = im_predicted.copy()

    for classname, list_cnt in zip(list(labeled_cnt.keys()),list(labeled_cnt.values())):
        for points in list_cnt:
            im_labeled_preview = cv.polylines(img=im_labeled_preview, pts=np.int32([points]), color=COLOUR[classname], isClosed=True, lineType=cv.LINE_4)
            im_labeled_preview = cv.addWeighted(
                im_labeled_preview, 1.0,
                im_labeled, .25,
                0.0
            )
            im_both_preview = cv.polylines(img=im_both_preview, pts=np.int32([points]), color=COLOUR[classname], isClosed=True, lineType=cv.LINE_4)
    for classname, list_cnt in zip(list(predicted_cnt.keys()),list(predicted_cnt.values())):
        for points in list_cnt:
            im_predicted_preview = cv.polylines(img=im_predicted, pts=np.int32([points]), color=COLOUR[classname], isClosed=True, lineType=cv.LINE_4)
            im_predicted_preview = cv.addWeighted(
                im_predicted_preview, 1.0,
                im_predicted, .25,
                0.0
            )
            #im_both_preview = cv.polylines(img=im_both_preview, pts=np.int32([points]), color=COLOUR[classname], isClosed=True, lineType=cv.LINE_4)
            im_both_preview = cv.addWeighted(
                im_both_preview, 1.0,
                im_predicted, .25,
                0.0
            )


    fig, axes = plt.subplots(1,3)
    axes[0].imshow(im_labeled_preview)
    axes[0].set_title("Labeled")
    axes[0].axis("off")

    axes[1].imshow(im_predicted_preview)
    axes[1].set_title("Predicted")    
    axes[1].axis("off")

    axes[2].imshow(im_both_preview)
    axes[2].set_title("Predicted+Labeled")    
    axes[2].axis("off")

    fig.suptitle(image_name)
    fig.tight_layout()
    plt.show()
