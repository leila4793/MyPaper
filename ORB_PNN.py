import numpy as np
import cv2
import glob
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from neupy import algorithms, environment


imageformat=".bmp"
path0 = "C:\\Users\\leila\\Desktop\\output\\patch_type3\\null_500"
path1 = "C:\\Users\\leila\\Desktop\\output\\patch_type3\\one_500"
path2 = "C:\\Users\\leila\\Desktop\\output\\patch_type3\\two_500"
path3 = "C:\\Users\\leila\\Desktop\\output\\patch_type3\\three_500"
imfilelist0 = [os.path.join(path0,f) for f in os.listdir(path0) if f.endswith(imageformat)]
imfilelist1 = [os.path.join(path1,f) for f in os.listdir(path1) if f.endswith(imageformat)]
imfilelist2 = [os.path.join(path2,f) for f in os.listdir(path2) if f.endswith(imageformat)]
imfilelist3 = [os.path.join(path3,f) for f in os.listdir(path3) if f.endswith(imageformat)]

name0 = os.listdir(path0)
name1 = os.listdir(path1)
name2 = os.listdir(path2)
name3 = os.listdir(path3)

data_path0 = os.path.join(path0)
data_path1 = os.path.join(path1)
data_path2 = os.path.join(path2)
data_path3 = os.path.join(path3)

files0 = glob.glob(data_path0)
files1 = glob.glob(data_path1)
files2 = glob.glob(data_path2)
files3 = glob.glob(data_path3)
class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 12.5, gamma = 0.50625):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):

        return self.model.predict(samples)[1].ravel()


print ('Calculating ORB descriptor')
orb = cv2.ORB_create(edgeThreshold=20, patchSize=20, nlevels=10 ,fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=200)

        

el0 = imfilelist0[0]
img0 = cv2.imread(el0)
img0 = cv2.resize(img0,(60,60))
kp = orb.detect(img0,None)
kp, des0 = orb.compute(img0, kp)
a0 = np.ones([1,50*32])
if np.any(des0):
    a00,a01 = des0.shape
    
else:
    a00 = 0
       
for i0 in range(a00):
    a0[0,i0*32:(i0+1)*32] = des0[i0]
    
descriptor0 = a0
imfilelist0.remove(imfilelist0[0])
for el0 in imfilelist0:
    img0 = cv2.imread(el0)
    img0 = cv2.resize(img0,(60,60))
    kp = orb.detect(img0,None)
    kp, des0 = orb.compute(img0, kp)
    if np.any(des0):
        a00,a01 = des0.shape
    else:
        a00 = 0
    for i0 in range(a00):
        a0[0,i0*32:(i0+1)*32] = des0[i0]
    descriptor0 = np.vstack([descriptor0,a0])


el1 = imfilelist1[0]
img1 = cv2.imread(el1)
img1 = cv2.resize(img1,(60,60))
kp = orb.detect(img1,None)
kp, des1 = orb.compute(img1, kp)
a1 = np.ones([1,50*32])
if np.any(des1):
    a10,a11 = des1.shape
else:
    a10 = 0
    
for i1 in range(a10):
    a1[0,i1*32:(i1+1)*32] = des1[i1]
    
descriptor1 = a1
imfilelist1.remove(imfilelist1[0])
for el1 in imfilelist1:
    des1 = None
    img1 = cv2.imread(el1)
    img1 = cv2.resize(img1,(60,60))
    kp = orb.detect(img1,None)
    kp, des1 = orb.compute(img1, kp)
    if np.any(des1):
        a10,a11 = des1.shape
    else:
        a10 = 0
    for i1 in range(a10):
        a1[0,i1*32:(i1+1)*32] = des1[i1]
    descriptor1 = np.vstack([descriptor1,a1])



el2 = imfilelist2[0]
img2 = cv2.imread(el2)
img2 = cv2.resize(img2,(60,60))
kp = orb.detect(img2,None)
kp, des2 = orb.compute(img2, kp)
a2 = np.ones([1,50*32])
if np.any(des2):
    a20,a21 = des2.shape
else:
    a20 = 0
    
for i2 in range(a20):
    a2[0,i2*32:(i2+1)*32] = des2[i2]
    
descriptor2 = a2
imfilelist2.remove(imfilelist2[0])
for el2 in imfilelist2:
    des2 = None
    img2 = cv2.imread(el2)
    img2 = cv2.resize(img2,(60,60))
    kp = orb.detect(img2,None)
    kp, des2 = orb.compute(img2, kp)
    if np.any(des2):
        a20,a21 = des2.shape
    else:
        a20 = 0
    for i2 in range(a20):
        a2[0,i2*32:(i2+1)*32] = des2[i2]
    descriptor2 = np.vstack([descriptor2,a2])


el3 = imfilelist3[0]
img3 = cv2.imread(el3)
img3 = cv2.resize(img3,(60,60))
kp = orb.detect(img3,None)
kp, des3 = orb.compute(img3, kp)
a3 = np.ones([1,50*32])
if np.any(des3):
    a30,a31 = des3.shape
else:
    a30 = 0
    
for i3 in range(a30):
    a3[0,i3*32:(i3+1)*32] = des3[i3]
    
descriptor3 = a3
imfilelist3.remove(imfilelist3[0])
for el3 in imfilelist3:
    des3 = None
    img3 = cv2.imread(el3)
    img3 = cv2.resize(img3,(60,60))
    kp = orb.detect(img3,None)
    kp, des3 = orb.compute(img3, kp)
    if np.any(des3):
        a30,a31 = des3.shape
    else:
        a30 = 0
    for i3 in range(a30):
        a3[0,i3*32:(i3+1)*32] = des3[i3]
    descriptor3 = np.vstack([descriptor3,a3])   


response0 = np.zeros([len(descriptor0),1])
response1 = np.ones([len(descriptor1),1])
response2 = 2*(np.ones([len(descriptor2),1]))
response3 = 3*(np.ones([len(descriptor3),1]))


descriptor = np.float32(np.vstack([descriptor0, descriptor1, descriptor2, descriptor3]))
response = np.int32(np.vstack([response0, response1, response2, response3]))


print ( 'Spliting data into training (80%) and test set (20%)')
msk = np.random.rand(len(descriptor)) < 0.8
traindata = np.float32(descriptor[msk])
traintarget1 = np.int32(response[msk])

testdata = np.float32(descriptor[~msk])
testtarget1 = np.int32(response[~msk])


traintarget =np.int32(np.squeeze(traintarget1))
testtarget =np.int32(np.squeeze(testtarget1))    
    
print ( 'Training PNN model')    
pnn = algorithms.PNN(std=10, verbose=False)
pnn.train(traindata, traintarget) 
print( 'Evaluating model')
resp = pnn.predict(testdata)
err = (testtarget != resp).mean()
print('Accuracy = ' + str(((1 - err)*100)))


