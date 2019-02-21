import cv2
import glob
import face_alignment
from skimage import io
import numpy as np
import pandas as pd
from numpy import ones, zeros, array
#from sklearn import preprocessing
from sklearn.decomposition import PCA as PCA
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from skimage.feature import hog as hog
from skimage import data, exposure
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import local_binary_pattern

def readimages():
    img=cv2.imread('C:\\Users\\MeaadAlrshoud\\Documents\\GitHub\\gender\\Female_Dataset\\29a.jpg')
    return img

def convertygrey(img):
    grey=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mat=np.array(np.float32(grey)) / 255.0
    return grey,mat

def normalization (img):
    normed= cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return normed

def haar(grey):
    face_cascade = cv2.CascadeClassifier('C:\\Users\\MeaadAlrshoud\\Documents\\GitHub\\gender\\haarcascade_frontalface_default.xml')
    # Detect faces in the image
    img = np.array(grey, dtype='uint8')
    f_haar= face_cascade.detectMultiScale(
    img,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in f_haar:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    return f_haar


def hog_features(l1):
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
    #hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
    #                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    # compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (8, 8)
    padding = (8, 8)
    locations = ((10, 20),)
    #f_hog = hog.compute(l1, winStride, padding, locations)
    fd, hog_image = hog(l1, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True)
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # initialize the HOG descriptor/person detecto
    return fd, hog_image_rescaled

def lbp(img):
    radius = 3
    n_points = 8 * radius
    lbp=local_binary_pattern(img, n_points, radius, method="uniform")
    return lbp

def face_alignment(img):
    fa= face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    preds = fa.get_landmarks(img)[-1]
    return preds


def main():
    img= readimages()
    img2,mat=convertygrey(img)
    #img3=normalization(img2)
    fd,img3=hog_features(img2)

    img4=haar(img2)
    im_lbp=lbp(img4)
    fa=face_alignment(img3)
    np.set_printoptions(threshold=np.inf)
    cv2.imshow('image 1',im_lbp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    np.savetxt("lbp.csv", img4, delimiter=",")
    print(img4)

if __name__ == "__main__":
	main()
