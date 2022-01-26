import glob, os
from random import random
from shutil import copyfile

DATA_PATH = r'D:/Konrad/TCM_scan/traning_segmentation/data'

val_precentage = 0.15 #15% 
test_precentage = 0.3 #15% (15-30)


def split_data():
    # Split data into train val and test
    os.chdir(DATA_PATH + r'/anot')
    for file in glob.glob("*.json"):
        split_name = file.split('.')[0]
        r = random()
        if(r < val_precentage): #VAL
            copyfile(DATA_PATH + r'/anot'+ '/'+ split_name + '.json',DATA_PATH + r'/val'+ '/'+ split_name + '.json')
            copyfile(DATA_PATH + r'/anot'+ '/'+ split_name + '.png',DATA_PATH + r'/val'+ '/'+ split_name + '.png')
            os.remove(DATA_PATH + r'/anot'+ '/'+ split_name + '.png')
            os.remove(DATA_PATH + r'/anot'+ '/'+ split_name + '.json')
        if(val_precentage < r < test_precentage): #TEST
            copyfile(DATA_PATH + r'/anot'+ '/'+ split_name + '.json',DATA_PATH + r'/test'+ '/'+ split_name + '.json')
            copyfile(DATA_PATH + r'/anot'+ '/'+ split_name + '.png',DATA_PATH + r'/test'+ '/'+ split_name + '.png')
            os.remove(DATA_PATH + r'/anot'+ '/'+ split_name + '.png')
            os.remove(DATA_PATH + r'/anot'+ '/'+ split_name + '.json')
        if(test_precentage < r): #TRAIN
            copyfile(DATA_PATH + r'/anot'+ '/'+ split_name + '.json',DATA_PATH + r'/train'+ '/'+ split_name + '.json')
            copyfile(DATA_PATH + r'/anot'+ '/'+ split_name + '.png',DATA_PATH + r'/train'+ '/'+ split_name + '.png')
            os.remove(DATA_PATH + r'/anot'+ '/'+ split_name + '.png')
            os.remove(DATA_PATH + r'/anot'+ '/'+ split_name + '.json')

def merge_data():
    # Before splitting put all data with annotations to the traing folder
    files = os.listdir(DATA_PATH + r'/val')
    for file in files:
        copyfile(DATA_PATH + r'/val'+ '/'+ file, DATA_PATH + r'/anot'+ '/'+ file)
        os.remove(DATA_PATH + r'/val'+ '/'+ file)
        
    files = os.listdir(DATA_PATH + r'/test')
    for file in files:
        copyfile(DATA_PATH + r'/test'+ '/'+ file, DATA_PATH + r'/anot'+ '/'+ file)
        os.remove(DATA_PATH + r'/test'+ '/'+ file)
    
    files = os.listdir(DATA_PATH + r'/train')
    for file in files:
        copyfile(DATA_PATH + r'/train'+ '/'+ file, DATA_PATH + r'/anot'+ '/'+ file)
        os.remove(DATA_PATH + r'/train'+ '/'+ file)
        

merge_data()
split_data()