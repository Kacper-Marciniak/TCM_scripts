import os
from shutil import copyfile

destination = r'D:\Konrad\TCM_scan\training_extraction\data\anot'
PATHES_LIST =  [
                r'D:\Konrad\TCM_scan\training_extraction\data\temp',] # Foldery do analizy, wpisz ile chesz - zrobi wszystkie jeden po drugim


for path_from in PATHES_LIST:
    
    files = os.listdir(path_from)

    for f in files:
        folder = path_from[path_from.rfind('\\')+1:]
        
        src = path_from  +'\\'+ str(f)
        dst = destination +'\\' + str(folder) +'_'+ str(f)
        print(src,'---------->',dst)
        copyfile(src,dst)
