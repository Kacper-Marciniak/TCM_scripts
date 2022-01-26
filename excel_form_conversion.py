import pandas as pd
import numpy as np

FILE = 'B_old.csv'
value = 'l'

EXCEL = r'D:\Konrad\TCM_scan\excel'
DASH = r'D:\Konrad\TCM_scan\dash'
df = pd.read_csv(DASH + '\\' + FILE) 
l_max = len(df['l_id'].unique())
w_max = len(df['w_id'].unique())
l_elements = df['l_id'].unique()
w_elements = df['w_id'].unique()
w_elements[::-1].sort()

tab = [[0 for col in range(w_max)] for row in range(l_max)]

for img_name in df['img_name']:
    wielkosc_stepienia = pd.to_numeric(df.loc[df['img_name'] == img_name, value])
    l = pd.to_numeric(df.loc[df['img_name'] == img_name, 'l_id'])
    w = pd.to_numeric(df.loc[df['img_name'] == img_name, 'w_id'])
    l, w, wielkosc_stepienia = np.array(l)[0], np.array(w)[0], np.array(wielkosc_stepienia)[0]    
    #print(l, w, wielkosc_stepienia)
    id_l = np.where(l_elements == l)[0][0]
    id_w = np.where(w_elements == w)[0][0]
    tab[id_l][id_w] = int(wielkosc_stepienia)

excel = pd.DataFrame(tab)
excel.to_csv (EXCEL +'//'+ FILE.split('.')[0] + '_' + value + '.csv', index = False, header = False)


