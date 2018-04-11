import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

def main():
    img_width = 150
    img_height = 150

    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
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

    model.load_weights('models/cnn-simple-moods.h5')

    img = image.load_img('./validation/sad.jpg', target_size=(150,150))

    x = image.img_to_array(img)

    x = np.divide(x, 255)

    x = np.expand_dims(x, axis=0)

    i = 1
    for pred in model.predict(x, 1):
        print("Prediction %d" % i)
        print("Afraid:%f %%" % (pred[0] * 100))
        print("Angry:%f %%" % (pred[1] * 100))
        print("Disgusted:%f %%" % (pred[2] * 100))
        print("Happy:%f %%" % (pred[3] * 100))
        print("Neutral:%f %%" % (pred[4] * 100))
        print("Sad:%f %%" % (pred[5] * 100))
        print("Suprised:%f %%" % (pred[6] * 100))

        i = i + 1

if __name__ == "__main__":
    main()