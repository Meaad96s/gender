from skimage.feature import local_binary_pattern
from skimage import data

radius = 3
n_points = 8 * radius
image = data.load('brick.png')
#print(image)
lbp = local_binary_pattern(image, n_points, radius, method="uniform")
print("Done!!")
print(lbp)
