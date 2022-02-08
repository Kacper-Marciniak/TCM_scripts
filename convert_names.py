
import pandas as pd
import numpy as np
import os

FILE = 'A_old.csv'
value = 'l'
PATH = r'D:\Konrad\TCM_scan\dash_skany\A_old'

EXCEL = r'D:\Konrad\TCM_scan\excel'
DASH = r'D:\Konrad\TCM_scan\dash'

df = pd.read_csv(DASH + '\\' + FILE) 
l_max = len(df['l_id'].unique())
w_max = len(df['w_id'].unique())
l_elements = df['l_id'].unique()
w_elements = df['w_id'].unique()
w_elements[::-1].sort()
l_elements[::-1].sort()


files = list(os.listdir(PATH + r'/images'))
for image_name in files: # Iterate over files
    base_name = image_name[:image_name.rfind('.')]
    split_name = base_name.split('_')
    row = int(split_name[1])
    col = int(split_name[0])
    id_l = np.where(l_elements == col)[0][0]
    id_w = np.where(w_elements == row)[0][0]
    new_name = str(id_l+1) + '_' + str(id_w+1) + '.png'
    print(image_name,'------>',new_name)
    os.rename(PATH + r'/images/' + image_name, PATH + r'/images/' + new_name)