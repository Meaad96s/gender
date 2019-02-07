import cv2
import glob
#import face_alignment
from skimage import io
import numpy as np
from numpy import ones, zeros, array
#from sklearn import preprocessing

#def main() :
    # Read RGB image
def readimages():
    #females = [cv2.imread(file) for file in glob.glob('C:\\Users\\MeaadAlrshoud\\Documents\\GitHub\\gender\\Female_Dataset\\*.jpg')]
    #males = [cv2.imread(file) for file in glob.glob('C:\\Users\\MeaadAlrshoud\\Documents\\GitHub\\gender\\Male_Dataset\\*.jpg')]
    females = [np.array(cv2.imread(file)) for file in glob.glob('C:\\Users\\MeaadAlrshoud\\Documents\\GitHub\\gender\\Female_Dataset\\*.jpg')]
    males = [np.array(cv2.imread(file)) for file in glob.glob('C:\\Users\\MeaadAlrshoud\\Documents\\GitHub\\gender\\Male_Dataset\\*.jpg')]
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
def haar(l1,l2):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Detect faces in the image
    f_haar= [ np.array(face_cascade.detectMultiScale(
    img,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
    )) for img in l1  ]


    m_haar= [ np.array(face_cascade.detectMultiScale(img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE)) for img in l2]


    return f_haar, m_haar



# Face Alignment
#def face_alignment(l1,l2):
#    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
#    input = io.imread('../test/assets/aflw-test.jpg')
#preds = fa.get_landmarks(m_haar)

def annotation (l1,l2):
    #ones= [1 for img in l1]
    a= ones([len(l1),1])
    #zeros=[0 for img in l2]
    #print(a)
    #np.append(l1,a)
    #l1.append(a)

    #print(l1)

    #np.append(l2,zeros,axis=1)
    return l1,l2




def main():
    females,males=readimages()
    females,males=convertygrey(females,males)
    #print(males)
    females,males=normalization(females,males) #all are 1
    #print(males)
    females,males=haar(females,males)
    print(males[0])
    #print("Found {0} faces!".format(len(females)))
    #print("Found {0} faces!".format(len(males)))
    #females,males=annotation(females,males)
    #print(females[0])

if __name__ == "__main__":
    main()
