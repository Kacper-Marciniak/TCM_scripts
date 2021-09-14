import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pandas as pd


DATA_FOLDER_PATH =  r'C:\Users\Konrad\tcm_scan\20210621_092043'

files = list(os.listdir(DATA_FOLDER_PATH))
print(len(files))
x,y,color,img_name=[],[],[],[]
for image_name in files:
    base_name = image_name[:image_name.rfind('.')]
    split_name = base_name.split('_')
    x.append(int(split_name[0]))
    y.append(int(split_name[1]))
    color.append(np.random.rand())
    img_name.append(str(image_name))
data = {'x':x,'y':y,'color':color,'img_name':img_name}

df = pd.DataFrame(data, columns= ['x', 'y','color','img_name'])  
df.to_csv (r'C:\Users\Konrad\tcm_scan\20210621_092043.csv', index = False, header=True)
