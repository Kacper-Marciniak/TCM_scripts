import os
from shutil import copyfile
from tkinter import filedialog
from tkinter import messagebox

#destination = r'D:\Konrad\TCM_scan\training_extraction\data\anot'
#PATHS_LIST =  [r'D:\Konrad\TCM_scan\training_extraction\data\temp',] # Foldery do analizy, wpisz ile chesz - zrobi wszystkie jeden po drugim

paths_input_list = list()

while True:
    paths_input_list.append(filedialog.askdirectory(title="Select path to input").replace('/','\\'))
    if not messagebox.askyesno(title="Add more folders?", message="Add more folders?"): break

destination = filedialog.askdirectory(title="Select path to output").replace('/','\\')


for path_input in paths_input_list:
    
    files = os.listdir(path_input)

    for file in files:
        folder = path_input.split('\\')[-1]
        src = os.path.join(path_input, str(file))
        dst = os.path.join(destination, f"{folder}_{file}")
        print(f"{src} -> {dst}")
        copyfile(src,dst)
