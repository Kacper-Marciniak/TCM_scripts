import os
import pandas as pd
import cv2 as cv

import numpy as np
import matplotlib.pyplot as plt

PATH = r'H:\Konrad\broach_comparison\A1'
PX2MM = 602.98


files = list(os.listdir(PATH))

df = []
for f in files:
    df.append(pd.read_csv(PATH + '\\' + f))


print(df[0][:3])
print(df[1][:3])

Px = []
Py = []
for tooth in range(len(df[0])):
    minx,miny,maxx,maxy = [],[],[],[]
    for broach in range(len(files)):
        minx.append(df[broach]['minx'][tooth])
        miny.append(df[broach]['miny'][tooth])
        maxx.append(df[broach]['maxx'][tooth])
        maxy.append(df[broach]['maxy'][tooth])

    Px.append((np.array(maxx)+np.array(minx))/2)
    Py.append((np.array(maxy)+np.array(miny))/2)


Px=(np.array(Px))/PX2MM
Py=(np.array(Py))/PX2MM

plt.grid()
plt.title('Rozrzut w osi X')
plt.xlabel('Numer zęba')
plt.ylabel('X [mm]')
plt.ylim(0,4)
plt.xlim(0,len(df[0]))
plt.plot(Px)
plt.show()

plt.grid()
plt.title('Rozrzut w osi Y')
plt.xlabel('Numer zęba')
plt.ylabel('Y [mm]')
plt.plot(Py)
plt.ylim(0,4)
plt.xlim(0,len(df[0]))
plt.show()


Px=Px.T
Py=Py.T
for i in range(len(files)):
    plt.scatter(Px[i],Py[i],s=20,label="skan " + str(i+1))
plt.grid()
plt.ylim(0,2748/PX2MM)
plt.xlim(0,3840/PX2MM)
plt.title('Położenie środków wykrytych zebów')
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.legend()
plt.show()

