import cv2
import glob
#import face_alignment
from skimage import io
from sklearn import preprocessing

#def main() :
    # Read RGB image
females = [cv2.imread(file) for file in glob.glob('C:\\Users\\FatenAldawish\\Documents\\GitHub\\gender\\Female_Dataset\\*.jpg')]
males = [cv2.imread(file) for file in glob.glob('C:\\Users\\FatenAldawish\\Documents\\GitHub\\gender\\Male_Dataset\\*.jpg')]

#convert to greyscale

f_grey= [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in females]
m_grey=[cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in males]
count=0
# print the images

# Normalization
normalized_f = [cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)for img in f_grey]
normalized_m =[cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)for img in m_grey]

# Haar
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Detect faces in the image
f_haar= [ face_cascade.detectMultiScale(
    img,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
    ) for img in normalized_f  ]

m_haar=[face_cascade.detectMultiScale(
    img,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
    ) for img in normalized_m]

# print haar affect
print("Found {0} faces!".format(len(f_haar)))
print("Found {0} faces!".format(len(m_haar)))

# Face Alignment

#fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

#input = io.imread('../test/assets/aflw-test.jpg')
#preds = fa.get_landmarks(m_haar)


    #if __name__ == "__main__":
    #    main()
    # execute only if run as a script
