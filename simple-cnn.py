#imports
# Dataset used: http://www.emotionlab.se/kdef/
# Authors: Sebastian Nilsson & Jonatan Nylund

import os

import tensorflow as tf
import keras
from PIL import ImageFile
from keras import backend as K
from keras import metrics
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():

    # Step 1 - Collect Data
    img_width, img_height = 150, 150
    train_data_dir = os.path.expanduser("moods-old/training")
    validation_data_dir = os.path.expanduser("moods-old/validation")

    # Rescale the input pixels
    datagen = ImageDataGenerator(rescale=1./255)

    # automagically retrieve images and their classes for train and validation sets
    train_generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=16)

    validation_generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=32)


    # Step 2 - Build Model
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))

    # To prevent overfiting we use Dropout
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    # Decrease Learning Rate Every Other Epoch

    from keras.callbacks import LearningRateScheduler
    def scheduler(epoch):
        if epoch%2==0 and epoch!=0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr*.9)
            print("lr changed to {}".format(lr*.9))
        return K.get_value(model.optimizer.lr)

    lr_decay = LearningRateScheduler(scheduler)

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', metrics.categorical_accuracy])

    nb_epoch = 16
    nb_train_samples = 4410
    nb_validation_samples = 490

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples / 16,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples / 16,
        verbose=1
    )

    model.save_weights('models/cnn-simple-moods-2.h5')

if __name__ == "__main__":
    main()