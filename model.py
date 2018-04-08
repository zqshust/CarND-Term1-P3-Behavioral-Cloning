# 1.Read in images
import csv

samples = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples.pop(0) #The first row is not path

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# 2. Generator
#use a generator to load data in batch_size portions to feed into Behavioral Cloning model
import numpy as np
import sklearn
from sklearn.utils import shuffle
from PIL import Image 
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])
                #Using Multiple Cameras
                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                # read in images from center, left and right cameras
                path = "../data/" # fill in the path to your training IMG directory
                img_center = np.array(Image.open(path+batch_sample[0]))
                img_left = np.array(Image.open(path+batch_sample[1]))
                img_right = np.array(Image.open(path+batch_sample[2]))
                
                images.extend([img_center, img_left, img_right]) 
                angles.extend([steering_center, steering_left, steering_right])
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

# 3. Setup the training model
# My model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
# Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
# NVIDIA's network consists of 9 layers, including a normalization layer, #5 convolutional layers and 3 fully connected layers.

ch, row, col = 3, 80, 320  # Trimmed image format

learning_rate=1e-4 # learning rate for optimizer
epochs = 3 # number of epochs to training the model
keep_prob = 0.5 # keep_prob for dropout layer

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.optimizers import Adam

model = Sequential()
#Data Preprocessing: Use a lambda layer to normalize input images
model.add(Lambda(lambda x: x/255.0-0.5, input_shape = (160,320,3)))
#Cropping Images: adding a cropping layer, choosing area of interest that excludes the sky & the hood of the car.
model.add(Cropping2D(cropping=((60,20), (0,0)))) 
model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
# Dropout layer with keep_prob = 0.5
model.add(Dropout(keep_prob))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# 4.Training the model
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose=0, 
                             save_best_only='true', mode='auto')
# Compile the model, use "mse"(mean squre error) as loss function to indicate how well the model predicts to the given steering angle for each image.
# Use Adam optimizer with lower learning rate 1e-4 rather than default 1e-3
#model.compile(loss='mse', optimizer='adam')
model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
#Traing the model
#model.fit(X_train,y_train, validation_split=0.2, shuffle=True, nb_epoch=epochs)
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*3, 
                    validation_data=validation_generator, nb_val_samples=len(validation_samples)*3, nb_epoch=epochs, verbose=1)
#Save the model I have trained.
model.save("model.h5")

#5. Visualizing Loss
import matplotlib.pyplot as plt
# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

exit()
