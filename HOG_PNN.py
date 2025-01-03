import numpy as np
import cv2
import glob
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from neupy import algorithms, environment


imageformat=".bmp"
path0 = "C:\\Users\\leila\\Desktop\\output\\type57_testresult\\null_500"
path1 = "C:\\Users\\leila\\Desktop\\output\\type57_testresult\\one_500"
path2 = "C:\\Users\\leila\\Desktop\\output\\type57_testresult\\two_500"
path3 = "C:\\Users\\leila\\Desktop\\output\\type57_testresult\\three_500"
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
    
print ('Calculating HoG descriptor')

def get_hog() : 
    winSize = (60,60)
    blockSize = (20,20)
    blockStride = (10,10)
    cellSize = (5,5)
    nbins = 9
    derivAperture = 1
    winSigma = -1
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)
    return hog
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR


hog = get_hog()
el = imfilelist0[0]
img0 = cv2.imread(el)
img0 = cv2.resize(img0,(60,60))
a0 = hog.compute(img0)
a01 , a02 = a0.shape
a0.resize(a02,a01)
descriptor0 = a0
imfilelist0.remove(imfilelist0[0])

for el0 in imfilelist0:
    img0 = cv2.imread(el0)
    img0 = cv2.resize(img0,(60,60))
    a0 = hog.compute(img0)
    a01 , a02 = a0.shape
    a0.resize(a02,a01)
    descriptor0 = np.vstack([descriptor0,a0])

el = imfilelist1[0]
img1 = cv2.imread(el)
img1 = cv2.resize(img1,(60,60))
a1 = hog.compute(img1)
a11 , a12 = a1.shape
a1.resize(a12,a11)
descriptor1 = a1
imfilelist1.remove(imfilelist1[0])

for el1 in imfilelist1:
    img1 = cv2.imread(el1)
    img1 = cv2.resize(img1,(60,60))
    a1 = hog.compute(img1)
    a11 , a12 = a1.shape
    a1.resize(a12,a11)
    descriptor1 = np.vstack([descriptor1,a1])


el = imfilelist2[0]
img2 = cv2.imread(el)
img2 = cv2.resize(img2,(60,60))
a2 = hog.compute(img2)
a21 , a22 = a2.shape
a2.resize(a22,a21)
descriptor2 = a2
imfilelist2.remove(imfilelist2[0])

for el2 in imfilelist2:
    img2 = cv2.imread(el2)
    img2 = cv2.resize(img2,(60,60))
    a2 = hog.compute(img2)
    a21 , a22 = a2.shape
    a2.resize(a22,a21)
    descriptor2 = np.vstack([descriptor2,a2])
    

el = imfilelist3[0]
img3 = cv2.imread(el)
img3 = cv2.resize(img3,(60,60))
a3 = hog.compute(img3)
a31 , a32 = a3.shape
a3.resize(a32,a31)
descriptor3 = a3
imfilelist3.remove(imfilelist3[0])

for el3 in imfilelist3:
    img3 = cv2.imread(el3)
    img3 = cv2.resize(img3,(60,60))
    a3 = hog.compute(img3)
    a31 , a32 = a3.shape
    a3.resize(a32,a31)
    descriptor3 = np.vstack([descriptor3,a3])

    
response0 = np.zeros([len(descriptor0),1])
response1 = np.ones([len(descriptor1),1])
response2 = 2*(np.ones([len(descriptor2),1]))
response3 = 3*(np.ones([len(descriptor3),1]))


descriptor = np.vstack([descriptor0, descriptor1, descriptor2, descriptor3])
descriptor = np.float32(descriptor)
response = np.vstack([response0, response1, response2, response3])
response = np.int32(response)

print ( 'Spliting data into training (80%) and test set (20%)')
msk = np.random.rand(len(descriptor)) < 0.8
traindata = np.float32(descriptor[msk])
testdata = np.float32(descriptor[~msk])
traintarget =np.int32(np.squeeze(response[msk]))
testtarget =np.int32(np.squeeze(response[~msk]))     
    
print ( 'Training PNN model')    
pnn = algorithms.PNN(std=10, verbose=False)
pnn.train(traindata, traintarget) 
print( 'Evaluating model')
resp = pnn.predict(testdata)
err = (testtarget != resp).mean()
print('Accuracy = ' + str(((1 - err)*100)))







    
    

    


    



