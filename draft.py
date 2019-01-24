


    '''
    for img in f_grey:
    # Output img with window name as 'image'
    cv2.imshow('image',img)
    # Maintain output window utill
    # user presses a key
    cv2.waitKey(0)
    # Destroying present windows on screen
    cv2.destroyAllWindows()
    '''

# Draw a rectangle around the faces
    for (x, y, w, h) in f_haar:
    f_rec = [cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) for img in females]
    m_rec = [cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) for img in males]

    cv2.imshow("Faces found", f_rec[0])
    cv2.waitKey(0)


# Face Alignment
    import face_alignment
from skimage import io
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

#input = io.imread('../test/assets/aflw-test.jpg')
    preds = fa.get_landmarks(m_haar)


# HOG

    import matplotlib.pyplot as plt

    from skimage.feature import hog
    from skimage import data, exposure

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np

    for img in f_grey:
    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Input image')

# Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
