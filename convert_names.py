import pandas as pd
import numpy as np
import os

FILE = 'test.csv'
PATH = r'D:\Konrad\TCM_scan\dash_skany\test'
DASH = r'D:\Konrad\TCM_scan\dash'

df = pd.read_csv(os.path.join(DASH.FILE)) 
l_elements = df['l_id'].unique()
w_elements = df['w_id'].unique()
w_elements[::-1].sort()
l_elements[::-1].sort()

print(l_elements)
print(w_elements)

files = list(os.listdir(os.path.join(PATH,'images')))
for image_name in files: # Iterate over files
    
    base_name = image_name.split('.')[-2]
    split_name = base_name.split('_')
    row = int(split_name[-1])
    col = int(split_name[-2])
    print(np.where(l_elements == col))
    id_l = np.where(l_elements == col)[0][0]
    id_w = np.where(w_elements == row)[0][0]
    print(image_name, id_l, id_w)
    new_name = f"{str(id_l+1)}_{str(id_w+1)}.png"
    print(f"{image_name} -> {new_name}")
    os.rename(f'{PATH}/images/{image_name}', f'{PATH}/images/{new_name}')