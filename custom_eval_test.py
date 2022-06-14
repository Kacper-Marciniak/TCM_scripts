import prepare_models
import os
import cv2 as cv
import numpy as np
from matplotlib.image import imread
import scipy.misc
from PIL import Image  
import glob
from matplotlib.image import imread
import json
import statistics

pth = r'D:\Konrad\TCM_scan\traning_segmentation\data\val\\'

models = prepare_models.Models()
segmentation_predictor = models.preapre_segmentation_model()

failures_dictionary = {
    0:"wykruszenie",
    1:"narost",
    2:"stepienie",
    3:"zatarcie"   
}
F1_score_global = []

for label in glob.glob(pth + '*json'):
    base_name = (label.split('\\')[-1]).split('.')[0]
    im = cv.imread(pth + base_name + '.png')
    #cv.imshow("test",im)
    file = open(label, 'r')
    json_label = json.load(file )

    label_outputs = {
        "wykruszenie": np.zeros_like(im),
        "narost": np.zeros_like(im),
        "stepienie": np.zeros_like(im),
        "zatarcie": np.zeros_like(im)
    }

    # Iterate over labels from .json file, merge it into 4 categories, draw it as bitmaps
    for shape in json_label["shapes"]:
        point_list = np.array(shape['points'])
        if len(point_list) > 2:
            pts = point_list.reshape((-1, 1, 2))  
            for class_name in failures_dictionary.values(): # Iterate over all instances classes
                if(shape["label"]==class_name):
                    label_outputs[class_name] = cv.fillPoly(label_outputs[class_name], np.int32([pts]), (255,255,255))
                    
    # Segmentation model inference
    predictions = segmentation_predictor(im) # Make prediction 
    pred_masks = predictions["instances"].to("cpu").pred_masks.numpy()
    pred_classes = predictions["instances"].to("cpu").pred_classes.numpy()
    num_instances = pred_masks.shape[0] 
    pred_masks = np.moveaxis(pred_masks, 0, -1)
    pred_masks_instance = []
    output = np.zeros_like(im)
    
    # Contains merged bitmaps for particular failures classes
    inference_outputs = {
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
                inference_outputs[failure_class] = np.where(pred_masks_instance[failure_class][-1] == True, 255, inference_outputs[failure_class])
 
    
    F1_score_mean = 0 #
    F1_counter = 0

    for class_name in failures_dictionary.values():
       
        pred = cv.cvtColor(inference_outputs[class_name], cv.COLOR_BGR2GRAY)
        label = cv.cvtColor(label_outputs[class_name], cv.COLOR_BGR2GRAY)
        bitwiseOr = cv.bitwise_or(pred, label)
        bitwiseAnd = cv.bitwise_and(pred, label) 
        bitwiseXor = cv.bitwise_xor(pred, label)
        TP = bitwiseAnd  # TP - a sample is predicted to be positive and its label is actually positive
        TN = cv.bitwise_not(bitwiseOr) # TN - a sample is predicted to be negative and its label is actually negative
        FP = cv.bitwise_xor(TP,pred) # FP - a sample is predicted to be positive and its label is actually negative
        FN = cv.bitwise_and(cv.bitwise_not(pred),cv.bitwise_xor(label,TP)) # FN - a sample is predicted to be negative and its label is actually positive

        TP = cv.countNonZero(TP)
        TN = cv.countNonZero(TN)
        FP = cv.countNonZero(FP)
        FN = cv.countNonZero(FN)
        acc = (TP+TN)/(TP+TN+FP+FN)
        precision = TP/(TP+FP) if (TP+FP) != 0 else 0
        recall = TP/(TP+FN) if (TP+FN) != 0 else 0
        F1_score = (2*precision*recall)/(precision+recall) if (precision+recall) != 0 else -1
       
        

        '''
        cv.imshow("pred" , pred)
        cv.imshow("label", label)
        cv.imshow("TP", TP)
        cv.imshow("TN", TN)
        cv.imshow("FP", FP)
        cv.imshow("FN", FN)
        '''
        
        #print("TP:",TP," TN:",TN, " FP:",FP, " FN:",FN)
        #print("Acc {} {:.2f}%".format(class_name,acc*100))
        #print("Precision {} {:.2f}%".format(class_name,precision*100))  
        #print("Recall {} {:.2f}%".format(class_name,recall*100))
          

        if F1_score != -1:
            print("F1 score {} {:.2f}%".format(class_name,F1_score*100))
            F1_score_mean += F1_score
            F1_counter += 1     
        
    if F1_counter != 0: F1_score_mean /= F1_counter                     
    print("F1 score {:.2f}%".format(F1_score_mean*100)) 
    F1_score_global.append(F1_score_mean)
    print(statistics.mean(F1_score_global)) 
    print('\n')
    #print(F1_score_global)
    
    #cv.waitKey(0)
print("Average F1 score:")
print(statistics.mean(F1_score_global))
