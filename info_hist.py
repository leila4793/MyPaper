import numpy as np
import cv2
import glob
import os
import time
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

N_points = 4000
n_bins = 200



imageformat=".bmp"
save_dir="D:/output_statistic/type57/New folder"
infoo = np.loadtxt("info57_700.txt",dtype={'names':('name', 'number of objects', 'x', 'y', 'w','h'),
                                 'formats':('|S30',np.float,np.float,np.float,np.float,np.float)})

imfilelistnew = []
name = []
xinfo = []
yinfo = []
winfo = []
hinfo = []
for i in range(len(infoo)):
    imfilelistnew.append('C:/Users/leila/Desktop/type57_700/training/positive/' + infoo[i][0].decode("utf-8"))
    name.append(infoo[i][0].decode("utf-8"))
    xinfo.append(int(infoo[i][2]))
    yinfo.append(int(infoo[i][3]))
    winfo.append(int(infoo[i][4]))
    hinfo.append(int(infoo[i][5]))


object_cascade = cv2.CascadeClassifier('type57win700.xml')
l = 0
for i in imfilelistnew:
    img = cv2.imread(i);
    objects = object_cascade.detectMultiScale(img)
    c = np.zeros([1,len(objects)+1])
    ii = 0
    for (x,y,w,h) in objects:
        c[0][ii] = w*h
        ii = ii+1
        
    ma = max(max(c))
    i1 = 0
    for i1 in range(len(c[0])-1):
        if c[0][i1] == ma:
            ob = i1
        i1 = i1+1
        
    if objects != ():
        finalobject = objects[ob]
        xi = finalobject[0]
        yi = finalobject[1]
        wi = finalobject[2]
        hi = finalobject[3]
        cv2.rectangle(img,(xi,yi),(xi+wi,yi+hi),(250,100,0),3)
    img1 = cv2.rectangle(img,(xinfo[l],yinfo[l]),(xinfo[l]+winfo[l],yinfo[l]+hinfo[l]),(120,100,0),3)
    cv2.imwrite(os.path.join(save_dir,name[l]),img1)
    l += 1
    
#fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
#axs[0].hist(w, bins=n_bins)
#axs[0].set_title('changes of w in dataset1 ( size = 150*250)')
#axs[1].hist(h, bins=n_bins)
#axs[1].set_title('changes of h in dataset1 ( size = 150*250)')
#plt.show(fig)

#fig1,axs1 = plt.subplots(1, 1, sharey=True, tight_layout=True)
#axs1[0].hist(len(infoo), bins=n_bins)
#plt.show(fig1)
