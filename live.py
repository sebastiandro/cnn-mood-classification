import numpy as np
import cv2
from keras.models import load_model
from densenet121 import densenet121_model
from keras.preprocessing import image
from custom_layers.scale_layer import Scale

cap = cv2.VideoCapture(0)

img_width, img_height = 299, 299  # Resolution of inputs
channel = 3
num_classes = 7
batch_size = 16

# Load our model
model = load_model('DenseNet121.h5', custom_objects={'Scale': Scale})
#model = None;
currentPrediction = "Neutral"

def predict(img):

    dst = cv2.resize(img, (299, 299), interpolation=cv2.INTER_CUBIC);
    x = image.img_to_array(dst)
    x = np.divide(x, 255)
    x = np.expand_dims(x, axis=0)
    i = 1

    for pred in model.predict(x, 1):

        max = 0
        maxJ = 0

        for j in range(0, len(pred)):
            if pred[j] > max:
                max = pred[j]
                maxJ = j

        if maxJ == 0:
            return "Afraid"
        elif maxJ == 1:
            return "Angry"
        elif maxJ == 2:
            return "Disgusted"
        elif maxJ == 3:
            return "Happy"
        elif maxJ == 4:
            return "Neutral"
        elif maxJ == 5:
            return "Sad"
        elif maxJ == 6:
            return "Suprised"

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    # Display the resulting frame
    # Write some Text

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (100,100)
    fontScale = 2
    fontColor = (0, 0, 255)
    lineType = 3

    currentPrediction = predict(frame)
    cv2.putText(gray, currentPrediction,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()