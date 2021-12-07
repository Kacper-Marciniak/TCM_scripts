import os
from shutil import copyfile

destination = r'D:\Konrad\TCM_scan\training_extraction\data\train'
PATHES_LIST =  [
                r'D:\Konrad\TCM_scan\Skany_nowe_pwr\pwr_c_3_20211001_133138',
                r'D:\Konrad\TCM_scan\Skany_nowe_pwr\pwr_b_3_20210930_151757',
                r'D:\Konrad\TCM_scan\Skany_nowe_pwr\pwr_a_2_20210930_104835'           
                ] # Foldery do analizy, wpisz ile chesz - zrobi wszystkie jeden po drugim


for path_from in PATHES_LIST:
    
    files = os.listdir(path_from + r'/images')

    for f in files:
        folder = path_from[path_from.rfind('\\')+1:]
        
        src = path_from  + r'/images/' + str(f)
        dst = destination +'\\' + str(folder) +'_'+ str(f)
        print(src,'---------->',dst)
        copyfile(src,dst)
