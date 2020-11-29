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
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Flatten, concatenate, Lambda, Input, Dropout, Dense, AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, Conv2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
from keras.utils import plot_model
from tensorflow.nn import local_response_normalization
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

# def get_model(weights='imagenet'):
#     fovea_input = Input(shape=(89,89,3), name='fovea_input')
#     x = Conv2D(96,11,3)(fovea_input)
#     x = Lambda(local_response_normalization,arguments={'alpha':1e-4,'bias':2})(x)
#     x = MaxPooling2D((2,2),padding='valid')(x)
#     x = Conv2D(256,5,1)(x)
#     x = Lambda(local_response_normalization,arguments={'alpha':1e-4,'bias':2})(x)
#     x = MaxPooling2D((2,2),padding='valid')(x)
#     x = Conv2D(384,1,1)(x)
#     x = Conv2D(384,1,1)(x)
#     x = Conv2D(256,3,1,activation='relu')(x)

#     context_input = Input(shape=(89,89,3), name='context_input')
#     c = Conv2D(96,11,3)(context_input)
#     c = Lambda(local_response_normalization,arguments={'alpha':1e-4,'bias':2})(c)
#     c = MaxPooling2D((2,2),padding='valid')(c)
#     c = Conv2D(256,5,1)(c)
#     c = Lambda(local_response_normalization,arguments={'alpha':1e-4,'bias':2})(c)
#     c = MaxPooling2D((2,2),padding='valid')(c)
#     c = Conv2D(384,1,1)(c)
#     c = Conv2D(384,1,1)(c)
#     c = Conv2D(256,3,1,activation='relu')(c)

#     cn = concatenate([x,c])

#     cn = Flatten()(cn)
#     fc = Dense(len(data.classes))(cn)
#     fc = Dropout(0.5)(fc)
#     predictions = Dense(len(data.classes),activation='softmax')(fc)

#     model = Model(inputs=[fovea_input, context_input], outputs=predictions)    
#     model.compile(optimizer=SGD(lr=0.0001, momentum=0.9) ,loss='categorical_crossentropy', metrics=['accuracy','top_k_categorical_accuracy'], run_eagerly=False)
    
#     return model

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
    weights_file = 'data/checkpoints/SF_MultiRes.1243-0.65.hdf5'
    main(weights_file)