import numpy as np
import cv2
import glob
import os
from sklearn.neighbors import KNeighborsClassifier


imageformat=".bmp"
path0 = "F:/clean_data/DB2/patch_null500"
path1 = "F:/clean_data/DB2/patch_one500"
path2 = "F:/clean_data/DB2/patch_two500"
path3 = "F:/clean_data/DB2/patch_three500"

pathtest0 = "C:/Users/leila/Desktop/outoutnew/patch_type5&7/null_300"
pathtest1 = "C:/Users/leila/Desktop/outoutnew/patch_type5&7/one_300"
pathtest2 = "C:/Users/leila/Desktop/outoutnew/patch_type5&7/two_300"
pathtest3 = "C:/Users/leila/Desktop/outoutnew/patch_type5&7/three_300"

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

print ('Calculating ORB descriptor')
orb = cv2.ORB_create(edgeThreshold=10, patchSize=5, nlevels=8 ,fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=50)

acc = 0
for i in range(1):
    el0 = imfilelist0[0]
    img0 = cv2.imread(el0)
    img0 = cv2.resize(img0,(100,50))
    kp = orb.detect(img0,None)
    kp, des0 = orb.compute(img0, kp)
    a0 = np.ones([1,55*32])
    if np.any(des0):
        a00,a01 = des0.shape
    
    else:
        a00 = 0
       
    for i0 in range(a00):
        a0[0,i0*32:(i0+1)*32] = des0[i0]


    
    descriptor0 = a0
    imfilelist0.remove(imfilelist0[0])
    for el0 in imfilelist0:
        des0 = None
        a0 = np.ones([1,55*32])
        img0 = cv2.imread(el0)
        img0 = cv2.resize(img0,(100,50))
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
    img1 = cv2.resize(img1,(100,50))
    kp = orb.detect(img1,None)
    kp, des1 = orb.compute(img1, kp)
    a1 = np.ones([1,55*32])
    if np.any(des1):
        a10,a11 = des1.shape
    else:
        a10 = 0
    
    for i1 in range(a10):
        a1[0,i1*32:(i1+1)*32] = des1[i1]
    
    descriptor1 = a1
    imfilelist1.remove(imfilelist1[0])
    for el1 in imfilelist1:
        a1 = np.ones([1,55*32])
        des1 = None
        img1 = cv2.imread(el1)
        img1 = cv2.resize(img1,(100,50))
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
    img2 = cv2.resize(img2,(100,50))
    kp = orb.detect(img2,None)
    kp, des2 = orb.compute(img2, kp)
    a2 = np.ones([1,55*32])
    if np.any(des2):
        a20,a21 = des2.shape
    else:
        a20 = 0
    
    for i2 in range(a20):
        a2[0,i2*32:(i2+1)*32] = des2[i2]
    
    descriptor2 = a2
    imfilelist2.remove(imfilelist2[0])
    for el2 in imfilelist2:
        a2 = np.ones([1,55*32])
        des2 = None
        img2 = cv2.imread(el2)
        img2 = cv2.resize(img2,(100,50))
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
    img3 = cv2.resize(img3,(100,50))
    kp = orb.detect(img3,None)
    kp, des3 = orb.compute(img3, kp)
    a3 = np.ones([1,55*32])
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
        a3 = np.ones([1,55*32])
        img3 = cv2.imread(el3)
        img3 = cv2.resize(img3,(100,50))
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
    descriptor = np.vstack([descriptor0, descriptor1, descriptor2, descriptor3])
    descriptor = np.float32(descriptor)

    response = np.vstack([response0, response1, response2, response3])
    response = np.int32(response)



#print ( 'Spliting data into training (70%) and test set (20%)')
    traindata = np.float32(descriptor)
    traintarget1 = np.int32(response)

    testdata = np.float32(descriptor)
    testtarget1 = np.int32(response)


    traintarget =np.int32(np.squeeze(traintarget1))
    testtarget =np.int32(np.squeeze(testtarget1))    
#print ( 'Training KNN model')    
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(traindata, traintarget) 
    #print( 'Evaluating model')
    resp = classifier.predict(testdata)
    err = (testtarget != resp).mean()
    acc = acc + (1 - err)*100
    
print('Accuracy = ' + str(acc))

