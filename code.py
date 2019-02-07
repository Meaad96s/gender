import cv2
import glob
#import face_alignment
from skimage import io
#from sklearn import preprocessing

#def main() :
    # Read RGB image
def readimages():
    females = [cv2.imread(file) for file in glob.glob('C:\\Users\\MeaadAlrshoud\\Documents\\GitHub\\gender\\Female_Dataset\\*.jpg')]
    males = [cv2.imread(file) for file in glob.glob('C:\\Users\\MeaadAlrshoud\\Documents\\GitHub\\gender\\Male_Dataset\\*.jpg')]
    return females,males



#convert to greyscale
def convertygrey(l1,l2):
    f_grey= [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in l1]
    m_grey=[cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in l2]
    return f_grey,m_grey


# print the images

# Normalization
def normalization (l1,l2):
    normalized_f = [cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)for img in l1]
    normalized_m =[cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)for img in l2]
    return normalized_f,normalized_m


# Haar
def haar(l1,l2):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Detect faces in the image
    f_haar= [ face_cascade.detectMultiScale(
    img,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
    ) for img in l1  ]

    m_haar=[face_cascade.detectMultiScale(img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE) for img in l2]

    return f_haar, m_haar



# Face Alignment

#fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

#input = io.imread('../test/assets/aflw-test.jpg')
#preds = fa.get_landmarks(m_haar)
def main():
    females,males=readimages()
    females,males=convertygrey(females,males)
    females,males=normalization(females,males)
    females,males=haar(females,males)
    print("Found {0} faces!".format(len(females)))
    print("Found {0} faces!".format(len(males)))

if __name__ == "__main__":
    main()
