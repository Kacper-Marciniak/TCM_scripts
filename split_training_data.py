import glob, os
import random
from shutil import copyfile

from PARAMETERS import PATH_TRAINING_DATA_SEGMENTATION
from tkinter_dialog_custom import askdirectory

#TODO: Improve splitting process

DATASET_PATH = askdirectory(title="Select dataset directory", initialdir=PATH_TRAINING_DATA_SEGMENTATION)

val_precentage = 0.15
test_precentage = 0.15


def split_data():
    # Split data into train val and test
    if not os.path.exists(os.path.join(DATASET_PATH,"anot")): os.makedirs(os.path.join(DATASET_PATH,"anot"))
    if not os.path.exists(os.path.join(DATASET_PATH,"val")): os.makedirs(os.path.join(DATASET_PATH,"val"))
    if not os.path.exists(os.path.join(DATASET_PATH,"test")): os.makedirs(os.path.join(DATASET_PATH,"test"))
    if not os.path.exists(os.path.join(DATASET_PATH,"train")): os.makedirs(os.path.join(DATASET_PATH,"train"))

    os.chdir(os.path.join(DATASET_PATH,'anot'))
    list_files = glob.glob("*.json")
    len_list_files = len(list_files)

    subset_lens = {
        "Test": round(test_precentage*len_list_files),
        "Val": round(val_precentage*len_list_files),
    }

    random.shuffle(list_files)

    list_files_test = list_files[0:subset_lens["Test"]]
    list_files_val = list_files[subset_lens["Test"]:subset_lens["Test"]+subset_lens["Val"]]
    list_files_train = list_files[subset_lens["Test"]+subset_lens["Val"]:]

    for i,file in enumerate(list_files_val):
        print(f"Val {i+1}/{len(list_files_val)}")
        split_name = file.split('.')[-2]
        copyfile(f'{DATASET_PATH}/anot/{split_name}.json', f'{DATASET_PATH}/val/{split_name}.json')
        copyfile(f'{DATASET_PATH}/anot/{split_name}.png', f'{DATASET_PATH}/val/{split_name}.png')
        os.remove(f'{DATASET_PATH}/anot/{split_name}.png')
        os.remove(f'{DATASET_PATH}/anot/{split_name}.json')
    for i,file in enumerate(list_files_test):
        print(f"Test {i+1}/{len(list_files_test)}")
        split_name = file.split('.')[-2]    
        copyfile(f'{DATASET_PATH}/anot/{split_name}.json',f'{DATASET_PATH}/test/{split_name}.json')
        copyfile(f'{DATASET_PATH}/anot/{split_name}.png',f'{DATASET_PATH}/test/{split_name}.png')
        os.remove(f'{DATASET_PATH}/anot/{split_name}.png')
        os.remove(f'{DATASET_PATH}/anot/{split_name}.json')
    for i,file in enumerate(list_files_train):
        print(f"Test {i+1}/{len(list_files_train)}")
        split_name = file.split('.')[-2]     
        copyfile(f'{DATASET_PATH}/anot/{split_name}.json',f'{DATASET_PATH}/train/{split_name}.json')
        copyfile(f'{DATASET_PATH}/anot/{split_name}.png',f'{DATASET_PATH}/train/{split_name}.png')
        os.remove(f'{DATASET_PATH}/anot/{split_name}.png')

def merge_data():
    # Before splitting put all data with annotations to the traing folder
    if not os.path.exists(os.path.join(DATASET_PATH,"anot")): os.makedirs(os.path.join(DATASET_PATH,"anot"))
    if not os.path.exists(os.path.join(DATASET_PATH,"val")): os.makedirs(os.path.join(DATASET_PATH,"val"))
    if not os.path.exists(os.path.join(DATASET_PATH,"test")): os.makedirs(os.path.join(DATASET_PATH,"test"))
    if not os.path.exists(os.path.join(DATASET_PATH,"train")): os.makedirs(os.path.join(DATASET_PATH,"train"))

    print('Merging data...')

    files = os.listdir(os.path.join(DATASET_PATH,'val'))
    for file in files:
        copyfile(f"{DATASET_PATH}/val/{file}", f"{DATASET_PATH}/anot/{file}")
        os.remove(f"{DATASET_PATH}/val/{file}")
        
    files = os.listdir(os.path.join(DATASET_PATH,'test'))
    for file in files:
        copyfile(f"{DATASET_PATH}/test/{file}", f"{DATASET_PATH}/anot/{file}")
        os.remove(f"{DATASET_PATH}/test/{file}")
    
    files = os.listdir(os.path.join(DATASET_PATH,'train'))
    for file in files:
        copyfile(f"{DATASET_PATH}/train/{file}", f"{DATASET_PATH}/anot/{file}")
        os.remove(f"{DATASET_PATH}/train/{file}")
        

merge_data()
split_data()

print("Finished splitting data!")