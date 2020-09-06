import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
import warnings
%matplotlib inline

img_generator = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, rescale=1/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# create the model
model = Sequential()

# add layers
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(150,150,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(150,150,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(150,150,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

# 2D to 1D so single dense layer can understand
model.add(Flatten())

model.add(Dense(128, activation='relu'))

# reduce overfitting by dropout
model.add(Dropout(rate=0.5))

# output layer 
model.add(Dense(1, activation='sigmoid'))

# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

### train the model ###
train_img_generator = img_generator.flow_from_directory('CATS_DOGS/train', target_size=(150,150), batch_size=16, class_mode='binary')

test_img_generator = img_generator.flow_from_directory('CATS_DOGS/test', target_size=(150,150), batch_size=16, class_mode='binary')

results = model.fit_generator(train_img_generator, epochs=100, steps_per_epoch=150, validation_data=test_img_generator, validation_steps=12)

# save the model
model.save("cat_dog_model.h5")

### predict ###
cat_image = 'CATS_DOGS/test/CAT/10003.jpg'

# resize image to size neural network is expecting
cat_image = image.load_img(cat_image, target_size=(150,150))

cat_image = image.img_to_array(cat_image)

cat_image = np.expand_dims(cat_image, axis=0)

cat_image = cat_image / cat_image.max()

model.predict_classes(cat_image) # 0 is cat and 1 is dog

model.predict(cat_image)
