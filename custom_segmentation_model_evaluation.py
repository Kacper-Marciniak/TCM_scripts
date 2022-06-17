import cv2 as cv
import os
import numpy as np
import json
import glob
import statistics
from PARAMETERS import  DICTIONARY_FAILURES

"""
Copy of old custom evaluation method
"""

class custom_metrics(): # structure to hold one instance of metrics
    def __init__(self) -> None:
        self.F1 = 0.0
        self.Recall = 0.0
        self.Precision = 0.0
        self.Accuracy = 0.0
        self.Specificity = 0.0

def custom_eval(segmentation_predictor,path_current_dataset):
    pth = os.path.join(path_current_dataset, r'test').replace('\\\\','\\').replace('\\','/')
    failures_dictionary = DICTIONARY_FAILURES
    
    F1_score_global = []
    Specificity_global = []
    Accuracy_global = []
    
    F1_score_global_stepienie, F1_score_global_narost, F1_score_global_zatarcie, F1_score_global_wykruszenie = [],[],[],[]

    for label in glob.glob(f'{pth}/*json'): # iterate through eval dataset
        base_name = (label.replace('\\\\','\\').replace('\\','/').split('/')[-1]).split('.')[0]
        file = open(label, 'r')
        json_label = json.load(file )
        im = cv.imread(os.path.join(pth,f"{base_name}.png"))
        print(os.path.join(pth,f"{base_name}.png"))
        pred_classes, image_metrics = calculate_metrics_for_image(im, json_label, segmentation_predictor)

        if list(DICTIONARY_FAILURES.keys())[list(DICTIONARY_FAILURES.values()).index("wykruszenie")] in pred_classes: 
            F1_score_global_wykruszenie.append(image_metrics["F1"]["wykruszenie"])
        if list(DICTIONARY_FAILURES.keys())[list(DICTIONARY_FAILURES.values()).index("narost")] in pred_classes: 
            F1_score_global_narost.append(image_metrics["F1"]["narost"])
        if list(DICTIONARY_FAILURES.keys())[list(DICTIONARY_FAILURES.values()).index("stepienie")] in pred_classes: 
            F1_score_global_stepienie.append(image_metrics["F1"]["stepienie"])
        if list(DICTIONARY_FAILURES.keys())[list(DICTIONARY_FAILURES.values()).index("zatarcie")] in pred_classes: 
            F1_score_global_zatarcie.append(image_metrics["F1"]["zatarcie"])

        Specificity_global.append(image_metrics["Specificity"])
        Accuracy_global.append(image_metrics["Accuracy"])
    
    F1_score_global= F1_score_global_narost+F1_score_global_stepienie+F1_score_global_wykruszenie+F1_score_global_zatarcie        
    
    if len(F1_score_global) > 0:
        return_F1_score_global = statistics.mean(F1_score_global)
    else: return_F1_score_global = -1
    if len(F1_score_global_stepienie) > 0:
        return_F1_score_global_stepienie = statistics.mean(F1_score_global_stepienie)
    else: return_F1_score_global_stepienie = -1
    if len(F1_score_global_narost) > 0:
        return_F1_score_global_narost = statistics.mean(F1_score_global_narost)
    else: return_F1_score_global_narost = -1
    if len(F1_score_global_zatarcie) > 0:
        return_F1_score_global_zatarcie = statistics.mean(F1_score_global_zatarcie)
    else: return_F1_score_global_zatarcie = -1
    if len(F1_score_global_wykruszenie) > 0:
        return_F1_score_global_wykruszenie = statistics.mean(F1_score_global_wykruszenie)
    else: return_F1_score_global_wykruszenie = -1

    return_accuracy_global = statistics.mean(Accuracy_global)
    return_specificity_global = statistics.mean(Specificity_global)

    print(
        f"""
        --- F1 ---
        Global:         {return_F1_score_global},
        Stępienie:      {return_F1_score_global_stepienie},
        Narost:         {return_F1_score_global_narost},
        Zatarcie:       {return_F1_score_global_zatarcie},
        Wykruszenie:    {return_F1_score_global_wykruszenie}
        """
    )

    F1_metrics = {
            "global":       return_F1_score_global,
            "stepienie":    return_F1_score_global_stepienie,
            "narost":       return_F1_score_global_narost,
            "zatarcie":     return_F1_score_global_zatarcie,
            "wykruszenie":  return_F1_score_global_wykruszenie
    }

    model_metrics = {
        "F1":                   F1_metrics,
        "Accuracy":             return_accuracy_global,
        "Specificity":          return_specificity_global,
    }

    return model_metrics

def calculate_metrics_for_image(im, json_label, segmentation_predictor):
        #cv.imshow("test",im)

        label_outputs = dict()
        for value in DICTIONARY_FAILURES.values():
            label_outputs[str(value)] = np.zeros_like(im)
        

        # Iterate over labels from .json file, merge it into 4 categories, draw it as bitmaps
        for shape in json_label["shapes"]:
            point_list = np.array(shape['points'])
            if len(point_list) > 2:
                pts = point_list.reshape((-1, 1, 2))  
                for class_name in DICTIONARY_FAILURES.values(): # Iterate over all instances classes
                    if(shape["label"]==class_name):
                        label_outputs[class_name] = cv.fillPoly(label_outputs[class_name], np.int32([pts]), (255,255,255))
                        
        # Segmentation model inference
        predictions = segmentation_predictor(im) # Make prediction 
        pred_masks = predictions["instances"].to("cpu").pred_masks.numpy()
        pred_classes = predictions["instances"].to("cpu").pred_classes.numpy()
        num_instances = pred_masks.shape[0] 
        pred_masks = np.moveaxis(pred_masks, 0, -1)
        pred_masks_instance = []
        
        # Contains merged bitmaps for particular failures classes
        inference_outputs = dict()
        for value in DICTIONARY_FAILURES.values():
            inference_outputs[str(value)] = np.zeros_like(im)

        # Contains temporary data used during merging
        pred_masks_instance = dict()
        for value in DICTIONARY_FAILURES.values():
            pred_masks_instance[str(value)] = []

        # Iterate over predicted defects, search for duplicated instances of the same class and merge them into single bitmap.
        for i in range(num_instances): 
            for class_id in DICTIONARY_FAILURES: # Iterate over all instances classes
                if(pred_classes[i] == class_id): 
                    failure_class = DICTIONARY_FAILURES[class_id] # Get name of the current class
                    pred_masks_instance[failure_class].append(pred_masks[:, :, i:(i+1)]) 
                    inference_outputs[failure_class] = np.where(pred_masks_instance[failure_class][-1] == True, 255, inference_outputs[failure_class])
    
        F1_score_stepienie_m, F1_score_wykruszenie_m, F1_score_zatarcie_m, F1_score_narost_m = 0,0,0,0
        accuracy_m, specificity_m = 0,0

        for class_name in DICTIONARY_FAILURES.values():
            # iterate through all potential failures
        
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

            accuracy = (TP+TN)/(TP+TN+FP+FN) # Accuracy is a measure of how many of the predictions were correct
            precision = TP/(TP+FP) if (TP+FP) != 0 else 0 # Precision is a measure of how many of the positive predictions made are correct (true positives)
            recall = TP/(TP+FN) if (TP+FN) != 0 else 0 # Recall is a measure of how many of the positive cases the classifier correctly predicted
            specificity = TN / (TN + FP) # Specificity is a measure of how many negative predictions made are correct (true negatives). 
            F1_score = (2*precision*recall) / (precision+recall) if (precision+recall) != 0 else -1

            if F1_score != -1:
                if class_name == "stepienie": F1_score_stepienie_m  += F1_score
                if class_name == "wykruszenie": F1_score_wykruszenie_m  += F1_score
                if class_name == "narost": F1_score_narost_m  += F1_score
                if class_name == "zatarcie": F1_score_zatarcie_m  += F1_score

            accuracy_m += accuracy
            specificity_m += specificity
      

        F1_metrics = {
            "stepienie": F1_score_stepienie_m,
            "narost": F1_score_narost_m,
            "zatarcie": F1_score_zatarcie_m,
            "wykruszenie": F1_score_wykruszenie_m
            }

        image_metrics = {
        "F1":                   F1_metrics,
        "Accuracy":             accuracy_m/4.0,
        "Specificity":          specificity_m/4.0
        }

        return pred_classes, image_metrics