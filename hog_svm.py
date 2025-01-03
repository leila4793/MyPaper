import numpy as np
import cv2
import glob
import os
from sklearn import svm
from numpy import matrix
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
from skimage import data, exposure
import skimage


imageformat=".bmp"
path0 = "F:/clean_data/DB2/patch_null700"
path1 = "F:/clean_data/DB2/patch_one700"
path2 = "F:/clean_data/DB2/patch_two700"
path3 = "F:/clean_data/DB2/patch_three700"

pathtest0 = "C:/Users/leila/Desktop/outoutnew/patch_type5&7/null_500"
pathtest1 = "C:/Users/leila/Desktop/outoutnew/patch_type5&7/one_500"
pathtest2 = "C:/Users/leila/Desktop/outoutnew/patch_type5&7/two_500"
pathtest3 = "C:/Users/leila/Desktop/outoutnew/patch_type5&7/three_500"

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

    
print ('Calculating HoG descriptor')
acc = 0
for i in range(5):
    descriptor0 = np.ones([len(imfilelist0),10000])
    l0 = 0
    for el0 in imfilelist0:
        img0 = cv2.imread(el0)
        img0 = cv2.resize(img0,(165,70))
        a0  = skimage.feature.hog(img0, orientations=12, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm=None,
                                  visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None)
        for i in range(len(a0)):
            descriptor0[l0,i] = a0[i]
        l0 = l0 + 1    
    descriptor1 = np.ones([len(imfilelist1),10000])
    l1 = 0
    for el1 in imfilelist1:
        img1 = cv2.imread(el1)
        img1 = cv2.resize(img1,(165,70))
        a1  = skimage.feature.hog(img1, orientations=12, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm=None,
                              visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None)
        for i in range(len(a1)):
            descriptor1[l1,i] = a1[i]
        l1 = l1 + 1 

    descriptor2 = np.ones([len(imfilelist2),10000])
    l2 = 0
    for el2 in imfilelist2:
        img2 = cv2.imread(el2)
        img2 = cv2.resize(img2,(165,70))
        a2  = skimage.feature.hog(img2, orientations=12, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm=None,
                              visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None)
        for i in range(len(a2)):
            descriptor2[l2,i] = a2[i]
        l2 = l2 + 1 
    
    descriptor3 = np.ones([len(imfilelist3),10000])
    l3 = 0
    for el3 in imfilelist3:
        img3 = cv2.imread(el3)
        img3 = cv2.resize(img3,(165,70))
        a3  = skimage.feature.hog(img3, orientations=12, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm=None,
                              visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None)
        for i in range(len(a3)):
            descriptor3[l3,i] = a3[i]
        l3 = l3 + 1

# test
    descriptortest0 = np.ones([len(imfilelisttest0),10000])
    l0 = 0
    for el0 in imfilelisttest0:
        img0 = cv2.imread(el0)
        img0 = cv2.resize(img0,(165,70))
        atest0  = skimage.feature.hog(img0, orientations=12, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm=None,
                              visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None)
    
        for i in range(len(atest0)):
            descriptortest0[l0,i] = atest0[i]
        l0 = l0 + 1 

        descriptortest1 = np.ones([len(imfilelisttest1),10000])
        l1 = 0
    for el1 in imfilelisttest1:
        img1 = cv2.imread(el1)
        img1 = cv2.resize(img1,(165,70))
        atest1  = skimage.feature.hog(img1, orientations=12, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm=None,
                            visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None)
    
        for i in range(len(atest1)):
            descriptortest1[l1,i] = atest1[i]
        l1 = l1 + 1 


    descriptortest2 = np.ones([len(imfilelisttest2),10000])
    l2 = 0
    for el2 in imfilelisttest2:
        img2 = cv2.imread(el2)
        img2 = cv2.resize(img2,(165,70))
        atest2  = skimage.feature.hog(img2, orientations=12, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm=None,
                            visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None)

        for i in range(len(atest2)):
            descriptortest2[l2,i] = atest2[i]
        l2 = l2 + 1 
    
    descriptortest3 = np.ones([len(imfilelisttest3),10000])
    l3 = 0
    for el3 in imfilelisttest3:
        img3 = cv2.imread(el3)
        img3 = cv2.resize(img3,(165,70))
        atest3  = skimage.feature.hog(img3, orientations=12, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm=None,
                            visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None)

        for i in range(len(atest3)):
            descriptortest3[l3,i] = atest3[i]
        l3 = l3 + 1 
    

    
    response0 = np.zeros([len(descriptor0),1])
    response1 = np.ones([len(descriptor1),1])
    response2 = 2*(np.ones([len(descriptor2),1]))
    response3 = 3*(np.ones([len(descriptor3),1]))

    responsetest0 = np.zeros([len(descriptortest0),1])
    responsetest1 = np.ones([len(descriptortest1),1])
    responsetest2 = 2*(np.ones([len(descriptortest2),1]))
    responsetest3 = 3*(np.ones([len(descriptortest3),1])) 


    descriptor = np.vstack([descriptor0, descriptor1, descriptor2, descriptor3])
    descriptor = np.float32(descriptor)

    response = np.vstack([response0, response1, response2, response3])
    response = np.int32(response)

    descriptortest = np.float32(np.vstack([descriptortest0, descriptortest1, descriptortest2, descriptortest3]))
    responsetest = np.int32(np.vstack([responsetest0, responsetest1, responsetest2, responsetest3]))

#print ( 'Spliting data into training (80%) and test set (20%)')
#msk = np.random.rand(len(descriptor)) < 0.8
    traindata = np.float32(descriptor)
    traintarget1 = np.int32(response)

    testdata = np.float32(descriptortest)
    testtarget1 = np.int32(responsetest)


    traintarget =np.int32(np.squeeze(traintarget1))
    testtarget =np.int32(np.squeeze(testtarget1))   
    
#print ( 'Training SVM model')    
    model = svm.SVC(kernel='linear') 
    model.fit(traindata, traintarget)
    model.score(traindata, traintarget)

    resp= model.predict(testdata)
    err = (testtarget != resp).mean()
    acc = (1 - err)*100
    print('Accuracy = ' + str(acc))








    
    

    


    



