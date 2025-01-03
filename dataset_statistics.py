import numpy as np
import cv2
import glob
import os
import time
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

N_points = 350
n_bins = 4

database1 = [215,335,349,332]
database2 = [156,164,180,188]

names = ['null' , 'one' , 'two' , 'three']

ax = plt.subplot(111)
ax.hist(database1)
plt.show(ax)
