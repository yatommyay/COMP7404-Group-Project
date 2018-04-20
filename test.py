#Refernce : https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
import os.path
import sys

classifier = Sequential()
if os.path.exists("model.h5"):
    print("Model is restored")
    classifier = load_model('comp7404model.h5')
    print("Loaded model from disk")
else:
    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # Part 2 - Fitting the CNN to the images

if (sys.argv[1] == "t"):	
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_datagen.flow_from_directory('dataset/training_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')
    test_set = test_datagen.flow_from_directory('dataset/test_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')
    classifier.fit_generator(training_set,
    steps_per_epoch = 52, #image size
    epochs = 10, #25
    validation_data = test_set,
    validation_steps = 100)
    # Part 3 - Making new predictions
    if os.path.exists('comp7404model.h5'):
       os.remove('comp7404model.h5')
    classifier.save('comp7404model.h5')
elif  (sys.argv[1] == "p"):
    if(len(sys.argv) > 2 and os.path.exists(sys.argv[2])):
     import numpy as np
     from keras.preprocessing import image
     test_image = image.load_img(sys.argv[2], target_size = (64, 64))
     test_image = image.img_to_array(test_image)
     test_image = np.expand_dims(test_image, axis = 0)
     result = classifier.predict(test_image)
     if result[0][0] == 1:
      prediction = 'happy'
     else:
      prediction = 'sad'
     print (prediction)
    else:
     print("Please input the correct image file path")
if (sys.argv[1] == "t"):	
    training_set.class_indices
