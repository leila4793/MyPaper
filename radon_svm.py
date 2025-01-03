import matplotlib.pyplot as plt
import cv2
import glob
import os
import numpy as np
from skimage.io import imread,imsave
from skimage import data_dir
from skimage.transform import radon, rescale
from scipy.ndimage import zoom
import warnings
import time
from sklearn import svm
from numpy import matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pickle


warnings.filterwarnings('ignore', '.*output shape of zoom.*')
warnings.filterwarnings('ignore', '.*Radon transform.*')

imageformat=".jpg"
path0 = "F:/clean_data/DB1/patchradon_null700"
path1 = "F:/clean_data/DB1/patchradon_one700"
path2 = "F:/clean_data/DB1/patchradon_two700"
path3 = "F:/clean_data/DB1/patchradon_three700"

pathtest0 = "C:/Users/leila/Desktop/radon/patchtype3_null700"
pathtest1 = "C:/Users/leila/Desktop/radon/patchtype3_one700"
pathtest2 = "C:/Users/leila/Desktop/radon/patchtype3_two700"
pathtest3 = "C:/Users/leila/Desktop/radon/patchtype3_three700"

imfilelist0 = [os.path.join(path0,f) for f in os.listdir(path0) if f.endswith(imageformat)]
imfilelist1 = [os.path.join(path1,f) for f in os.listdir(path1) if f.endswith(imageformat)]
imfilelist2 = [os.path.join(path2,f) for f in os.listdir(path2) if f.endswith(imageformat)]
imfilelist3 = [os.path.join(path3,f) for f in os.listdir(path3) if f.endswith(imageformat)]

imfilelisttest0 = [os.path.join(pathtest0,f) for f in os.listdir(pathtest0) if f.endswith(imageformat)]
imfilelisttest1 = [os.path.join(pathtest1,f) for f in os.listdir(pathtest1) if f.endswith(imageformat)]
imfilelisttest2 = [os.path.join(pathtest2,f) for f in os.listdir(pathtest2) if f.endswith(imageformat)]
imfilelisttest3 = [os.path.join(pathtest3,f) for f in os.listdir(pathtest3) if f.endswith(imageformat)]

name0 = os.listdir(path0)
name1 = os.listdir(path1)
name2 = os.listdir(path2)
name3 = os.listdir(path3)

nametest0 = os.listdir(pathtest0)
nametest1 = os.listdir(pathtest1)
nametest2 = os.listdir(pathtest2)
nametest3 = os.listdir(pathtest3)

data_path0 = os.path.join(path0)
data_path1 = os.path.join(path1)
data_path2 = os.path.join(path2)
data_path3 = os.path.join(path3)

data_pathtest0 = os.path.join(pathtest0)
data_pathtest1 = os.path.join(pathtest1)
data_pathtest2 = os.path.join(pathtest2)
data_pathtest3 = os.path.join(pathtest3)

files0 = glob.glob(data_path0)
files1 = glob.glob(data_path1)
files2 = glob.glob(data_path2)
files3 = glob.glob(data_path3)

filestest0 = glob.glob(data_pathtest0)
filestest1 = glob.glob(data_pathtest1)
filestest2 = glob.glob(data_pathtest2)
filestest3 = glob.glob(data_pathtest3)

def sortrow(img):
    a,b = img.shape
    outrow = np.squeeze(img.reshape((1,a*b)))
    return(outrow)

def sortcol(img):
    a,b = img.shape
    c = np.zeros((b,a))
    for i in range(b):
        c[i] = matrix(img).transpose()[i].getA()[0]

    outcol = np.int32(np.squeeze(np.reshape(c,(1,a*b))))
    return(outcol)


acc = 0
for i in range(200):
    el0 = imfilelist0[0]
    img0 = cv2.imread(el0,0)
    descriptor0 = sortcol(img0)
    imfilelist0.remove(imfilelist0[0])
    for el0 in imfilelist0:
        img0 = cv2.imread(el0,0)
        a0 = sortcol(img0)
        descriptor0 = np.vstack([descriptor0,a0])

    el1 = imfilelist1[0]
    img1 = cv2.imread(el1,0)
    descriptor1 = sortcol(img1)
    imfilelist1.remove(imfilelist1[0])
    for el1 in imfilelist1:
        img1 = cv2.imread(el1,0)
        a1 = sortcol(img1)
        descriptor1 = np.vstack([descriptor1,a1])


    el2 = imfilelist2[0]
    img2 = cv2.imread(el2,0)
    descriptor2 = sortcol(img2)
    imfilelist2.remove(imfilelist2[0])
    for el2 in imfilelist2:
        img2 = cv2.imread(el2,0)
        a2 = sortcol(img2)
        descriptor2 = np.vstack([descriptor2,a2])    
    
    el3 = imfilelist3[0]
    img3 = cv2.imread(el3,0)
    descriptor3 = sortcol(img3)
    imfilelist3.remove(imfilelist3[0])
    for el3 in imfilelist3:
        img3 = cv2.imread(el3,0)
        a3 = sortcol(img3)
        descriptor3 = np.vstack([descriptor3,a3])


#test
    el0 = imfilelisttest0[0]
    img0 = cv2.imread(el0,0)
    descriptortest0 = sortcol(img0)
    imfilelisttest0.remove(imfilelisttest0[0])
    for el0 in imfilelisttest0:
        img0 = cv2.imread(el0,0)
        a0 = sortcol(img0)
        descriptortest0 = np.vstack([descriptortest0,a0])

    el1 = imfilelisttest1[0]
    img1 = cv2.imread(el1,0)
    descriptortest1 = sortcol(img1)
    imfilelisttest1.remove(imfilelisttest1[0])
    for el1 in imfilelisttest1:
        img1 = cv2.imread(el1,0)
        a1 = sortcol(img1)
        descriptortest1 = np.vstack([descriptortest1,a1])

    el2 = imfilelisttest2[0]
    img2 = cv2.imread(el2,0)
    descriptortest2 = sortcol(img2)
    imfilelisttest2.remove(imfilelisttest2[0])
    for el2 in imfilelisttest2:
        img2 = cv2.imread(el2,0)
        a2 = sortcol(img2)
        descriptortest2 = np.vstack([descriptortest2,a2])

    el3 = imfilelisttest3[0]
    img3 = cv2.imread(el3,0)
    descriptortest3 = sortcol(img3)
    imfilelisttest3.remove(imfilelisttest3[0])
    for el3 in imfilelisttest3:
        img3 = cv2.imread(el3,0)
        a3 = sortcol(img3)
        descriptortest3 = np.vstack([descriptortest3,a3])

    

    response0 = np.zeros([len(descriptor0),1])
    response1 = np.ones([len(descriptor1),1])
    response2 = 2*(np.ones([len(descriptor2),1]))
    response3 = 3*(np.ones([len(descriptor3),1]))

    responsetest0 = np.zeros([len(descriptortest0),1])
    responsetest1 = np.ones([len(descriptortest1),1])
    responsetest2 = 2*(np.ones([len(descriptortest2),1]))
    responsetest3 = 3*(np.ones([len(descriptortest3),1])) 

    #responsetest0 = np.zeros([len(descriptor0),1])
    #responsetest1 = np.ones([len(descriptor1),1])
    #responsetest2 = 2*(np.ones([len(descriptor2),1]))
    #responsetest3 = 3*(np.ones([len(descriptor3),1]))

    descriptor = np.vstack([descriptor0, descriptor1, descriptor2, descriptor3, descriptortest0, descriptortest1, descriptortest2, descriptortest3])
    descriptor = np.float32(descriptor)

    response = np.vstack([response0, response1, response2, response3, responsetest0, responsetest1, responsetest2, responsetest3])
    response = np.int32(response)

    descriptortest = np.float32(np.vstack([descriptortest0, descriptortest1, descriptortest2, descriptortest3]))
    responsetest = np.int32(np.vstack([responsetest0, responsetest1, responsetest2, responsetest3]))

    print ( 'Spliting data into training (80%) and test set (20%)')
    msk = np.random.rand(len(descriptor)) < 0.8
    traindata = np.float32(descriptor[msk])
    traintarget1 = np.int32(response[msk])

    testdata = np.float32(descriptor[~msk])
    testtarget1 = np.int32(response[~msk])


    traintarget =np.int32(np.squeeze(traintarget1))
    testtarget =np.int32(np.squeeze(testtarget1))   
    
    #print ( 'Training SVM model')    

    model = svm.SVC(kernel='poly', C=12.5, gamma=0.50625) 
    model.fit(traindata, traintarget)
    s = pickle.dumps(clf,'D:/project/raja')
    model.score(traindata, traintarget)


    resp= model.predict(testdata)
    err = (testtarget != resp).mean()
    acc = acc + (1 - err)*100
    
#print('Accuracy = ' + str(((1 - err)*100)))
print('Accuracy = ' + str(acc))
