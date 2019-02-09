import cv2
import glob
# import face_alignment
from skimage import io
import numpy as np
import pandas as pd
from numpy import ones, zeros, array
from skimage.feature import local_binary_pattern
#from sklearn import preprocessing
from sklearn.decomposition import PCA as PCA
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# def main() :
# Read RGB image
def readimages():
	# females = [cv2.imread(file) for file in glob.glob('C:\\Users\\MeaadAlrshoud\\Documents\\GitHub\\gender\\Female_Dataset\\*.jpg')]
	# males = [cv2.imread(file) for file in glob.glob('C:\\Users\\MeaadAlrshoud\\Documents\\GitHub\\gender\\Male_Dataset\\*.jpg')]
	females = [cv2.imread(file) for file in
			   glob.glob('C:\\Users\\MeaadAlrshoud\\Documents\\GitHub\\gender\\Female_Dataset\\*.jpg')]
	males = [cv2.imread(file) for file in
			 glob.glob('C:\\Users\\MeaadAlrshoud\\Documents\\GitHub\\gender\\Male_Dataset\\*.jpg')]
	return females, males


# convert to greyscale
def convertygrey(l1, l2):
	# Applying Grayscale filter to image
	f_grey = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in l1]
	m_grey = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in l2]
	f_grey = [np.array(np.float32(img)) / 255.0 for img in f_grey]
	m_grey = [np.array(np.float32(img)) / 255.0 for img in m_grey]
	return f_grey, m_grey


# print the images

# Normalization
def normalization(l1, l2):
	normalized_f = [cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) for img in l1]
	normalized_m = [cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) for img in l2]
	return normalized_f, normalized_m


# Haar face detetction
def haar(l1, l2):
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	# Detect faces in the image
	f_haar = [np.array(face_cascade.detectMultiScale(
		img,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)) for img in l1]

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x, y, w, h) in faces:
		img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
		roi_gray = gray[y:y + h, x:x + w]
		roi_color = img[y:y + h, x:x + w]

	m_haar = [np.array(face_cascade.detectMultiScale(img,
													 scaleFactor=1.1,
													 minNeighbors=5,
													 minSize=(30, 30),
													 flags=cv2.CASCADE_SCALE_IMAGE)) for img in l2]
	return f_haar, m_haar


# Face Alignment
# def face_alignment(l1,l2):
#    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
#    input = io.imread('../test/assets/aflw-test.jpg')
# preds = fa.get_landmarks(m_haar)

# def spatial-scale():

def annotation(l1, l2):
	# ones= [1 for img in l1]
	label1 = ones((len(l1), 1))
	label2= zeros((len(l2), 1))
	#zeros=[0 for img in l2]
	# print(a)
	np.append(l1,label1)
	# l1.append(a)
	np.append(l2,label2,axis=1)
	return l1, l2


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

def dr_pca(x_train):
	# X_std = StandardScaler().fit_transform(X)
	# mean_vec = np.mean(X_std, axis=0)
	# cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
	# print('Covariance matrix \n%s' %cov_mat)
	# Y = sklearn_pca.fit_transform(X_std)

	# find the principal components
	N_COMPONENTS = 2
	pca = PCA(n_components=N_COMPONENTS, random_state=0, svd_solver='randomized')
	X_train_pca = pca.fit_transform(x_train)
	# X_test_pca = pca.transform(X_test)

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


def main():
	females, males = readimages()
	females, males = convertygrey(females, males)
	# print(males)
	females, males = normalization(females, males)  # all are 1
	# print(males)
	# females,males=haar(females,males)
	# print(males)
	# print("Found {0} faces!".format(len(females)))
	# print("Found {0} faces!".format(len(males)))
	females, males = annotation(females, males)
	print(females[1])



	females = hog(females)
	print(females[1])
    females,males=lbp(females,males)
    #print(males)

	females = dr_pca(females)
	# X = dataset.iloc[:, 0:4].values  #features
	# y = dataset.iloc[:, 4].values #labels
	# X_train, X_test, y_train, y_test = train_test_split(females,y,test_size=0.2,random_state=0)
	# X_train = StandardScaler().fit_transform(X_train)
	# X_test = StandardScaler().fit_transform(X_test)

	'''classifier = RandomForestClassifier(max_depth=2, random_state=0)
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)
	cm = confusion_matrix(y_test, y_pred)
	print(cm)
	print('Accuracy' + str(accuracy_score(y_test, y_pred)))'''


###LDA:
# females,males= LDA(females,males)

# randomly order the data
# seed(0)
# shuffle(raw_data)

if __name__ == "__main__":
	main()
