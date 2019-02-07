def extract_features(image_paths):
    """
    extract_features computed the inception bottleneck feature for a list of images

    image_paths: array of image path
    return: 2-d array in the shape of (len(image_paths), 2048)
    """

    feature_dimension = 4096
    features = np.empty((len(image_paths), feature_dimension))

    for i, image_path in enumerate(image_paths):

        if not gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        else :
            print('Processing %s...' % (image_path))
        img = image.load_img(image_path,target_size=(224, 224))
        '''
        img = image.load_img(image_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        feature = model.predict(img_data)
        features[i, :] = np.squeeze(feature)
        '''
        img_input = np.expand_dims(img,0)

        #feature = vgg19.predict(np.flip(img_input-RGB_MEAN_PIXELS, axis=-1))
        features[i, :] = np.squeeze(feature)

    return features
