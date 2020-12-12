"""
Test on images split into directories. This assumes we've split
our videos into frames and moved them to their respective folders
and trained our model.

Based on:
https://keras.io/preprocessing/image/
and
https://keras.io/applications/
"""
from train_custom_cnn import get_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, concatenate, Lambda, Input, Dropout, Dense, MaxPooling2D, Conv2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from data import DataSet
import time
import os.path

data = DataSet()

# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath=os.path.join('data', 'checkpoints', 'inception.{epoch:03d}-{accuracy:.2f}.hdf5'),
    verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir=os.path.join('data', 'logs'))

# Helper: Save results.
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('data', 'logs', 'InceptionV3' + '-' + 'training-' + \
    str(timestamp) + '.log'))

def crop_center(img):
    cropx = 89
    cropy = 89
    y,x,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]-117

def subtract_mean(img):
    return img-117

def get_test_generator():
    context_datagen = ImageDataGenerator()
    
    fovea_datagen = ImageDataGenerator(
    preprocessing_function=crop_center)

    context_generator = context_datagen.flow_from_directory(
        os.path.join('data', 'test'),
        target_size=(89, 89),
        batch_size=32,
        classes=data.classes,
        class_mode='categorical',
        interpolation='bilinear')

    fovea_generator = fovea_datagen.flow_from_directory(
        os.path.join('data', 'test'),
        target_size=(89, 89),
        batch_size=32,
        classes=data.classes,
        class_mode='categorical')

    while True:
        f = fovea_generator.next()
        c = context_generator.next()
        yield [f[0],c[0]], f[1]


def test_model(model, test_generator, nb_steps, callbacks=[]):
    #train_generator, validation_generator = generators
    model.evaluate(
        test_generator,
        verbose=1,
        steps=nb_steps,
        callbacks=callbacks)
    return model

def main(weights_file):
    model = get_model()
    print(model.summary())

    test_generator = get_test_generator()
    
    # if weights_file is None:
    #     print("Loading network from ImageNet weights.")
    #     # Get and train the top layers.
    #     model = freeze_all_but_top(model)
    #     model = train_model(model, 10, generators)
    # else:
    print("Loading saved model: %s." % weights_file)
    model.load_weights(weights_file)

    callbacks = [checkpointer, tensorboard, csv_logger]
    model = test_model(model, test_generator, 100)

if __name__ == '__main__':
    weights_file = 'data/checkpoints/SF_MultiRes.1233-0.79.hdf5'
    main(weights_file)