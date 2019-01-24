import cv2
import glob

#def main() :
    # Read RGB image
females = [cv2.imread(file) for file in glob.glob('C:\\Users\\MeaadAlrshoud\\Documents\\GitHub\\Gender-Classification\\Female_Dataset\\*.jpg')]
males = [cv2.imread(file) for file in glob.glob('C:\\Users\\MeaadAlrshoud\\Documents\\GitHub\\Gender-Classification\\Male_Dataset\\*.jpg')]

#convert to greyscale

f_grey= [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in females]
m_grey=[cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in males]
count=0
# print the images

# Normalization



# Haar
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Detect faces in the image
f_haar= [ face_cascade.detectMultiScale(
    img,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
    ) for img in f_grey]

m_haar=[face_cascade.detectMultiScale(
    img,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
    ) for img in m_grey]

# print haar affect
print("Found {0} faces!".format(len(f_haar)))
print("Found {0} faces!".format(len(m_haar)))

# Face Alignment
import face_alignment
from skimage import io
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

#input = io.imread('../test/assets/aflw-test.jpg')
preds = fa.get_landmarks(m_haar)


    #if __name__ == "__main__":
    #    main()
    # execute only if run as a script
