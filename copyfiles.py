import os
from shutil import copyfile
import random

from tkinter_dialog_custom import askdirectory
from tkinter_dialog_custom import askYesNo

#destination = r'D:\Konrad\TCM_scan\training_extraction\data\anot'
#PATHS_LIST =  [r'D:\Konrad\TCM_scan\training_extraction\data\temp',] # Foldery do analizy, wpisz ile chesz - zrobi wszystkie jeden po drugim

def list_paths_to_all_files(path):
    list_files = list()
    for element in os.listdir(path):
        if os.path.isdir(os.path.join(path,element)):
            files = list_paths_to_all_files(os.path.join(path,element))
            for file in files:
                list_files.append(file)
        else:
            list_files.append(os.path.join(path,element))
    return list_files

paths_input_list = list()

NUMBER_OF_FILES_TO_COPY = 100 # <------------------------------------

while True:
    paths_input_list.append(askdirectory(title="Select path to input").replace('/','\\'))
    if not askYesNo(title="Add more folders?", message="Add more folders?"): break

destination = askdirectory(title="Select path to output").replace('/','\\')

list_paths_files = []
for path_input in paths_input_list:
    for path in list_paths_to_all_files(path_input):
        list_paths_files.append(os.path.join(path_input,path))

if NUMBER_OF_FILES_TO_COPY < len(list_paths_files):
    list_paths_files = random.sample(list_paths_files,NUMBER_OF_FILES_TO_COPY) 

    
for path in list_paths_files:
    folder = path.split('\\')[-2]
    name  = path.split('\\')[-1]
    src = path
    dst = os.path.join(destination, f"{folder}_{name}")
    print(f"{src} -> {dst}")
    copyfile(src,dst)
