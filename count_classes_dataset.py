"""
Counting iterations of each class inside the dataset
"""

import os, json
from tkinter_dialog_custom import askdirectory
from PARAMETERS import PATH_TRAINING_DATA_SEGMENTATION
from PARAMETERS import DICTIONARY_FAILURES

def get_path_to_dataset():
    while True:
        path_dataset = askdirectory(title="Select dataset folder", initialdir=PATH_TRAINING_DATA_SEGMENTATION)
        if os.path.exists(path_dataset): break
    return path_dataset

def count_classes_in_dataset(path_dataset):
    path_train = os.path.join(path_dataset,r"annotations/data_train.json").replace('\\\\','\\').replace('\\','/')
    path_val = os.path.join(path_dataset,r"annotations/data_val.json").replace('\\\\','\\').replace('\\','/')
    path_test = os.path.join(path_dataset,r"annotations/data_test.json").replace('\\\\','\\').replace('\\','/')

    results = dict()

    for i,pth in enumerate([path_train,path_val,path_test]):
        results[pth.split('/')[-1].replace('.json','')] = count_json_labels(pth)

    dict_classes = {
        "wykruszenie":0,
        "narost":0,
        "stepienie":0,
        "zatarcie":0   
    }

    for i,class_name in enumerate(list(results.values())[0].keys()): # iterating through failure classes
        dict_classes[class_name] = sum([subset[class_name] for subset in results.values()]) # calculate sum for entire dataset

    results["Sum for entire dataset"] = dict_classes
    
    return results

def count_json_labels(path_json):
    file = open(path_json, 'r')
    json_label = json.load(file )

    dict_classes = {
        "wykruszenie":0,
        "narost":0,
        "stepienie":0,
        "zatarcie":0   
    }

    for annot in json_label["annotations"]:
        class_id = annot["category_id"]
        if class_id == list(DICTIONARY_FAILURES.keys())[list(DICTIONARY_FAILURES.values()).index("wykruszenie")]: 
            dict_classes["wykruszenie"] += 1
        if class_id == list(DICTIONARY_FAILURES.keys())[list(DICTIONARY_FAILURES.values()).index("narost")]: 
            dict_classes["narost"] += 1
        if class_id == list(DICTIONARY_FAILURES.keys())[list(DICTIONARY_FAILURES.values()).index("stepienie")]: 
            dict_classes["stepienie"] += 1
        if class_id == list(DICTIONARY_FAILURES.keys())[list(DICTIONARY_FAILURES.values()).index("zatarcie")]: 
            dict_classes["zatarcie"] += 1
    
    
    return dict_classes

def print_results(results):
    for i,subset in enumerate(results.values()):
        print(f"Inside {list(results.keys())[i]}:")
        for class_name, class_count in zip(subset.keys(),subset.values()):
            print(f"    * {class_name} -> {class_count}")
    

def save_results_file(path_dataset, results):
    path = os.path.join(path_dataset,r"class_count.json")
    file = open(path, "w")
    if file.closed: return -1 # ERROR
    json.dump(results, file, indent=4)
    file.close()

pth = get_path_to_dataset()
results = count_classes_in_dataset(pth)
print_results(results)
save_results_file(pth, results)