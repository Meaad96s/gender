import cv2
import glob
#import face_alignment
from skimage import io
import numpy as np
import pandas as pd
from numpy import ones, zeros, array, transpose
from skimage.feature import local_binary_pattern
#from sklearn import preprocessing & PCA
from sklearn.decomposition import PCA as PCA
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
# import SVM
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split
import pandas as pd


#def main() :
    # Read RGB image
def readimages():
    females = [cv2.imread(file) for file in glob.glob('C:\\Users\\fatenAldawish\\Documents\\GitHub\\gender\\Female_Dataset\\*.jpg')]
    males = [cv2.imread(file) for file in glob.glob('C:\\Users\\fatenAldawish\\Documents\\GitHub\\gender\\Female_Dataset\\*.jpg')]
    return females,males



#convert to greyscale
def convertygrey(l1,l2):
    f_grey= [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in l1]
    m_grey=[cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in l2]
    return f_grey,m_grey


# print the images

# Normalization
def normalization (l1,l2):
    normalized_f = [np.array(cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)) for img in l1]
    normalized_m =[np.array(cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)) for img in l2]
    return normalized_f,normalized_m


# Haar face detetction
def haar(l1, l2):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Detect faces in the image
    ll1 = np.array(l1, dtype='uint8')
    ll2= np.array(l2,dtype='uint8')
    f_haar = [np.array(face_cascade.detectMultiScale(img,
                                                     scaleFactor=1.1,
                                                     minNeighbors=5,
                                                     minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)) for img in ll1]

    m_haar = [np.array(face_cascade.detectMultiScale(img,
                                                     scaleFactor=1.1,
                                                     minNeighbors=5,
                                                     minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)) for img in ll2]
    return f_haar, m_haar

def spatial_scale(l1,l2):
    f_scale = scale(l1)
    m_scale = scale(l2)
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(l1)
    y = sc_y.fit_transform(l2)
    return X,y
# Face Alignment
#def face_alignment(l1,l2):
#    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
#    input = io.imread('../test/assets/aflw-test.jpg')
# preds = fa.get_landmarks(m_haar)

# def spatial-scale():

def annotation(l1, l2):

    f = [np.expand_dims(file,0) for file in l1]
    m = [np.expand_dims(file,0) for file in l2]
    #ones= [1 for img in l1]
    #print(len(ones))
    print("L1")
    #print(l1)
    label1 = ones((len(l1), 1),dtype='float')

    #label2= zeros((len(l2), 1),dtype='float')
    #zeros=[0 for img in l2]
    print(label1.shape)
    print(l1)
    np.column_stack((l1,label1))
    # l1.append(a)
    #np.append(np.atleast_3d(l1), label1, axis=1).shape
    data=np.append(l1,label1,axis=1)
    #np.append(l2,label2,axis=1)
    #print(label2.shape,l1.shape)
    return data


def hog(l1):
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    # compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (8, 8)
    padding = (8, 8)
    locations = ((10, 20),)
    l1 = np.array(l1, dtype='uint8')
    f_hog = [hog.compute(img, winStride, padding, locations) for img in l1]
    # Rescale histogram for better display
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return f_hog

def lbp(l1,l2):
    radius = 3
    n_points = 8 * radius
    f_lbp= [local_binary_pattern(img, n_points, radius, method="uniform") for img in l1]
    m_lbp=[local_binary_pattern(img, n_points, radius, method="uniform")for img in l2]
    return f_lbp,m_lbp

def dr_pca(x,t):
    # X_std = StandardScaler().fit_transform(X)
    # mean_vec = np.mean(X_std, axis=0)
    # cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
    # print('Covariance matrix \n%s' %cov_mat)
    # Y = sklearn_pca.fit_transform(X_std)

    # find the principal components
    N_COMPONENTS = 4
    pca = PCA(n_components=N_COMPONENTS, random_state=0, svd_solver='randomized')
    X_train_pca = pca.fit_transform(x)
    print(t.shape)
    X_test_pca = pca.transform(t)
    return X_train_pca,X_test_pca
    '''pcaclf = clf.fit(X_train_pca, y_train)
    print("pca test score", pcaclf.score(X_test_pca, y_test))
    print("pca train score", pcaclf.score(X_train_pca, y_train))'''
    # X_r = pca.fit(X).transform(X)
    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))

    # def LDA():
    '''females,males= LDA(females,males)
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    ldaclf = clf.fit(X_train_lda, y_train)
    print("lda train score", ldaclf.score(X_train_lda, y_train))
    print("lda test score", ldaclf.score(X_test_lda, y_test))'''




def svm(X_train, X_test, y_train, y_test):
    clf=SVC(gamma='auto')
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    scores = cross_val_score(clf, X_test, y_test, cv=10)
    print (scores)
    return scores

def experiment1():
    females, males = readimages()
    females, males = convertygrey(females, males)
    f = np.asarray(females, dtype=np.float32)
    m = np.asarray(males, dtype=np.float32)

    f,m=haar(f,m)
    f = np.asarray(f, dtype=np.float32)
    m = np.asarray(m, dtype=np.float32)
    #print(f)
    f=hog(f)
    m=hog(m)
    #nsamples, nx, ny = f.shape
    #d2_train_dataset = f.reshape((nsamples,nx*ny))
    #females= annotation(d2_train_dataset)
    label1 = ones((len(f), 1),dtype='float')
    label2= zeros((len(m), 1),dtype='float')
    #np.append(np.atleast_3d(l1), label1, axis=1).shape
    l=np.vstack((label1,label2))
    d=np.vstack((f,m))
    #data=np.append(f,m)
    #X = pd.DataFrame(data)
    dataset_size = len(d)
    TwoDim_dataset = d.reshape(dataset_size,-1)
    y = l
    #print(y.shape,TwoDim_dataset.shape )
    X_train, X_test, y_train, y_test = train_test_split(TwoDim_dataset,y,test_size=0.2,random_state=0)
    pca_train,pca_test=dr_pca(X_train,X_test)
    score = svm(pca_train, pca_test, y_train, y_test)




def main():
    experiment1()
    # print(males)
    #females, males = normalization(females, males)  # all are 1
    # print(males)
    # females,males=haar(females,males)
    # print(males)
    # print("Found {0} faces!".format(len(females)))
    # print("Found {0} faces!".format(len(males)))





if __name__ == "__main__":
    main()
