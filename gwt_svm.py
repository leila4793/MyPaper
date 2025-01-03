import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

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

def gw(w,theta,r,c,du,dv):
    a = np.array(np.zeros((r,c)), dtype = complex)
    for i in range(r):
        for j in range(c):
            u = (i*np.cos(theta)) + (j*np.sin(theta))
            v = (-i*np.sin(theta)) + (j*np.cos(theta))
            fu = (1/(np.sqrt(du)))*np.exp(-(u**2)/(2*(du)**2))
            fv = (1/(np.sqrt(dv)))*np.exp(-(v**2)/(2*(dv)**2))
            a[i][j] = (fu)*(fv)*(np.exp(np.complex(0,w*u)))
            
    return(a)


def mean(numbers):
    meani = float(sum(numbers)) / max(len(numbers), 1)
    return (meani)

def meannumber(mat):
    row = len(mat);
    col = len(mat[0]);
    outmat = [];
    for i in range(row):
        outmat.append(mean(mat[i]))
    outma = mean(outmat)
    return(outma)

def meanimage(x):
    row,col = x.shape;
    outmean = [];
    for i in range(row):
      outmean.append(mean(x[i]))
    outp = mean(outmean)  
    return(outp)

def normimage(x):
    r,c = np.shape(np.array(x))
    poweri = np.sqrt(np.sum(np.multiply(x,x)))
    return(poweri)

def ixy (x,v):
    m = meanimage(x)
    var = np.var(x.var(0))
    a = (v/var) * (x - m*(np.ones(np.shape(x))))
    return(a)

#def normwindow(image, stepSize):
    #tmp = image;
    #for x in range(0, image.shape[1], stepSize):
        #for y in range(0, image.shape[0], stepSize):
            #window = image[x:x + stepSize, y:y + stepSize]
            ##cv2.rectangle(tmp, (x, y), (x + stepSize, y + stepSize), (255,125,100));
            ##cv2.imshow('img',cv2.rectangle(tmp, (x, y), (x + stepSize, y + stepSize), (255,125,100)))
            #vi = np.var(window);
            #meani = meanimage(window)*(np.squeeze(np.ones(window.shape)));
            #v = normimage(image,stepSize);
            #normi = (v/vi)*(window - meani);
            #tmp[x:x + stepSize, y:y + stepSize] = normi
    #return(tmp)

def awa(x,g):
    aw = np.sum(np.multiply(x,g))/(normimage(g));
    return(aw)

def Awa(awaa,psize):
    aw1 = np.real(awaa)
    return(aw1)


def main (x,r,c,w,theta,du,dv,v):
    lw = len(w)
    ltheta = len(theta)
    t = np.squeeze(np.zeros((1,lw*ltheta),dtype = complex))
    l = 0
    for i in range(lw):
        for j in range(ltheta):
            gwx = gw(w[i],theta[j],r,c,du,dv)
            ix = ixy(x,v)
            aw = Awa(awa(ix,gwx),r)
            t[l] = aw
            l = l+1
    return(t)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


w =[8,12,16,20]
theta = [-5,0,5]
dv = 12
du = 16
v = 25
r = 32
c =32


el = imfilelist0[0]
img0 = cv2.imread(el,0)
img0 = cv2.resize(img0,(32,32))
a0 = main(img0,r,c,w,theta,dv,du,v)
descriptor0 = a0
imfilelist0.remove(imfilelist0[0])

for el0 in imfilelist0:
    img0 = cv2.imread(el0,0)
    img0 = cv2.resize(img0,(32,32))
    a0 = main(img0,r,c,w,theta,dv,du,v)
    descriptor0 = np.vstack([descriptor0,a0])

el = imfilelist1[0]
img1 = cv2.imread(el,0)
img1 = cv2.resize(img1,(32,32))
a1 = main(img1,r,c,w,theta,dv,du,v)
descriptor1 = a1
imfilelist1.remove(imfilelist1[0])

for el1 in imfilelist1:
    img1 = cv2.imread(el1,0)
    img1 = cv2.resize(img1,(32,32))
    a1 = main(img1,r,c,w,theta,dv,du,v)
    descriptor1 = np.vstack([descriptor1,a1])


el = imfilelist2[0]
img2= cv2.imread(el,0)
img2 = cv2.resize(img2,(32,32))
a2 = main(img2,r,c,w,theta,dv,du,v)
descriptor2 = a2
imfilelist2.remove(imfilelist2[0])

for el2 in imfilelist2:
    img2 = cv2.imread(el2,0)
    img2 = cv2.resize(img2,(32,32))
    a2 = main(img2,r,c,w,theta,dv,du,v)
    descriptor2 = np.vstack([descriptor2,a2])
    

el = imfilelist3[0]
img3 = cv2.imread(el,0)
img3 = cv2.resize(img3,(32,32))
a3 = main(img3,r,c,w,theta,dv,du,v)
descriptor3 = a3
imfilelist3.remove(imfilelist3[0])

for el3 in imfilelist3:
    img3 = cv2.imread(el3,0)
    img3 = cv2.resize(img3,(32,32))
    a3 = main(img3,r,c,w,theta,dv,du,v)
    descriptor3 = np.vstack([descriptor3,a3])

# test
el = imfilelisttest0[0]
img0 = cv2.imread(el,0)
img0 = cv2.resize(img0,(32,32))
a0 = main(img0,r,c,w,theta,dv,du,v)
descriptortest0 = a0
imfilelisttest0.remove(imfilelisttest0[0])

for el0 in imfilelisttest0:
    img0 = cv2.imread(el0,0)
    img0 = cv2.resize(img0,(32,32))
    a0 = main(img0,r,c,w,theta,dv,du,v)
    descriptortest0 = np.vstack([descriptortest0,a0])

el = imfilelisttest1[0]
img1 = cv2.imread(el,0)
img1 = cv2.resize(img1,(32,32))
a1 = main(img1,r,c,w,theta,dv,du,v)
descriptortest1 = a1
imfilelisttest1.remove(imfilelisttest1[0])

for el1 in imfilelisttest1:
    img1 = cv2.imread(el1,0)
    img1 = cv2.resize(img1,(32,32))
    a1 = main(img1,r,c,w,theta,dv,du,v)
    descriptortest1 = np.vstack([descriptortest1,a1])


el = imfilelisttest2[0]
img2= cv2.imread(el,0)
img2 = cv2.resize(img2,(32,32))
a2 = main(img2,r,c,w,theta,dv,du,v)
descriptortest2 = a2
imfilelisttest2.remove(imfilelisttest2[0])

for el2 in imfilelisttest2:
    img2 = cv2.imread(el2,0)
    img2 = cv2.resize(img2,(32,32))
    a2 = main(img2,r,c,w,theta,dv,du,v)
    descriptortest2 = np.vstack([descriptortest2,a2])
    

el = imfilelisttest3[0]
img3 = cv2.imread(el,0)
img3 = cv2.resize(img3,(32,32))
a3 = main(img3,r,c,w,theta,dv,du,v)
descriptortest3 = a3
imfilelisttest3.remove(imfilelisttest3[0])

for el3 in imfilelisttest3:
    img3 = cv2.imread(el3,0)
    img3 = cv2.resize(img3,(32,32))
    a3 = main(img3,r,c,w,theta,dv,du,v)
    descriptortest3 = np.vstack([descriptortest3,a3])


        
response0 = np.zeros([len(descriptor0),1])
response1 = np.ones([len(descriptor1),1])
response2 = 2*(np.ones([len(descriptor2),1]))
response3 = 3*(np.ones([len(descriptor3),1]))

responsetest0 = np.zeros([len(descriptortest0),1])
responsetest1 = np.ones([len(descriptortest1),1])
responsetest2 = 2*(np.ones([len(descriptortest2),1]))
responsetest3 = 3*(np.ones([len(descriptortest3),1])) 

descriptor = np.float32(np.vstack([descriptor0, descriptor1, descriptor2, descriptor3]))
response = np.int32(np.vstack([response0, response1, response2, response3]))

descriptortest = np.float32(np.vstack([descriptortest0, descriptortest1, descriptortest2, descriptortest3]))
responsetest = np.int32(np.vstack([responsetest0, responsetest1, responsetest2, responsetest3]))

print ( 'Spliting data into training (80%) and test set (20%)')
#msk = np.random.rand(len(descriptor)) < 0.8
traindata = np.float32(descriptor)
traintarget1 = np.int32(response)

testdata = np.float32(descriptortest)
testtarget1 = np.int32(responsetest)


traintarget =np.int32(np.squeeze(traintarget1))
testtarget =np.int32(np.squeeze(testtarget1))     

model = svm.SVC(kernel='rbf', C=12.5, gamma=0.50625) 
model.fit(traindata, traintarget)
model.score(traindata, traintarget)

resp= model.predict(testdata)
err = (testtarget != resp).mean()
print('Accuracy = ' + str(((1 - err)*100)))







            



            
    







        
