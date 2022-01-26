import os
from shutil import copyfile

destination = r'D:\Konrad\TCM_scan\traning_segmentation\data\train'
PATHES_LIST =  [
                r'D:\Konrad\TCM_scan\dash_skany\C_old',
                r'D:\Konrad\TCM_scan\dash_skany\D_new',
         
                ] # Foldery do analizy, wpisz ile chesz - zrobi wszystkie jeden po drugim


for path_from in PATHES_LIST:
    
    files = os.listdir(path_from + r'/otsu_tooth')

    for f in files:
        folder = path_from[path_from.rfind('\\')+1:]
        
        src = path_from  + r'/otsu_tooth/' + str(f)
        dst = destination +'\\' + str(folder) +'_'+ str(f)
        print(src,'---------->',dst)
        copyfile(src,dst)
