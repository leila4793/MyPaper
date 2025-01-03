import numpy as np
import cv2
import glob
import os
import time


imageformat=".bmp"
save_dir="D:/output_statistic/type57/New folder"
infoo = np.loadtxt("info57_700.txt",dtype={'names':('name', 'number of objects', 'x', 'y', 'w','h'),
                                 'formats':('|S30',np.float,np.float,np.float,np.float,np.float)})

imfilelistnew = []
name = []
x = []
y = []
w = []
h = []
for i in range(len(infoo)):
    imfilelistnew.append('C:/Users/leila/Desktop/type57_700/training/positive/rawdata' + infoo[i][0].decode("utf-8"))
    name.append(infoo[i][0].decode("utf-8"))
    x.append(int(infoo[i][2]))
    y.append(int(infoo[i][3]))
    w.append(int(infoo[i][4]))
    h.append(int(infoo[i][5]))


l = 0
for i in imfilelistnew:
    img = cv2.imread(i);
    img1 = cv2.rectangle(img,(x[l],y[l]),(x[l]+w[l],y[l]+h[l]),(120,100,0),3)
    cv2.imwrite(os.path.join(save_dir,name[l]),img1)
    l += 1

