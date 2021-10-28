import os
import pandas as pd
import cv2 as cv

import numpy as np
import matplotlib.pyplot as plt

PATH = r'D:\Konrad\TCM_scan\broach_comparison\RR'
PX2MM = 602.98

files = list(os.listdir(PATH))

df = []
for f in files:
    df.append(pd.read_csv(PATH + '\\' + f))


print(df[0][:3])


Px = []
Py = []
Lx,Ly=[],[]
for tooth in range(len(df[0])):
    minx,miny,maxx,maxy = [],[],[],[]
    for broach in range(len(files)):
        minx.append(df[broach]['minx'][tooth])
        miny.append(df[broach]['miny'][tooth])
        maxx.append(df[broach]['maxx'][tooth])
        maxy.append(df[broach]['maxy'][tooth])

    Lx.append((np.array(maxx)-np.array(minx)))
    Ly.append((np.array(maxy)-np.array(miny)))
    Px.append((np.array(maxx)+np.array(minx))/2)
    Py.append((np.array(maxy)+np.array(miny))/2)



Lx=(np.array(Lx))/PX2MM
Ly=(np.array(Py))/PX2MM

plt.grid()
plt.title('Rozrzut w osi X')
plt.xlabel('Numer zęba')
plt.ylabel('X [mm]')
plt.ylim(0,4)
plt.xlim(0,len(df[0]))
plt.plot(Lx)
plt.show()

plt.grid()
plt.title('Rozrzut w osi Y')
plt.xlabel('Numer zęba')
plt.ylabel('Y [mm]')
plt.plot(Ly)
plt.ylim(0,4)
plt.xlim(0,len(df[0]))
plt.show()


Lx=Lx.T
Ly=Ly.T
for i in range(len(files)):
    print(type(Px[i]))
    plt.scatter(Lx[i].mean(),Ly[i].mean(),s=20,label="skan " + str(i+1))
    plt.Circle((Lx[i].mean(), Ly[i].mean()), 0.5, color ='r')

plt.grid()
plt.ylim(0,2748/PX2MM)
plt.xlim(0,3840/PX2MM)
plt.title('Położenie środków wykrytych zebów')
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.legend()
plt.show()

