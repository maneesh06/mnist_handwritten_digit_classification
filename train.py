from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.models import model_from_json
from keras.datasets import mnist
import os

## environment variable
MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
WEIGHTS_FILE = os.environ["WEIGHTS_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
WEIGHTS_PATH = os.path.join(MODEL_DIR, WEIGHTS_FILE)

## load the dataset
(X_train, y_train), (X_val, y_val) = mnist.load_data()

## reshape the input image of size (28, 28, 1) and create batch of it
x_train = X_train.reshape(-1, 28, 28, 1)
x_val = X_val.reshape(-1, 28, 28, 1)

x_train = x_train.astype("float32")/255.
x_val = x_val.astype("float32")/255.

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

## create a Sequential model
model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu', input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

## create augmented image
datagen = ImageDataGenerator(
    zoom_range = 0.1, 
    height_shift_range = 0.1, 
    width_shift_range = 0.1, 
    rotation_range = 10
)

## compile the model with loss metrics
model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

hist = model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=16),
    steps_per_epoch=500,
    epochs=25,
    verbose=2, 
    validation_data=(x_val[:400,:], y_val[:400,:]),
    callbacks=[annealer]

)

## save weights
model.save(WEIGHTS_PATH)
print("model weights saved in model.h5 file")

## save model
model_json = model.to_json()
with open(MODEL_PATH, "w") as json_file:
    json_file.write(model_json)
print("model saved as model.json file")
