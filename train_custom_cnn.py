"""
Train on images split into directories. This assumes we've split
our videos into frames and moved them to their respective folders.

Based on:
https://keras.io/preprocessing/image/
and
https://keras.io/applications/
"""
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Flatten, concatenate, Lambda, Input, Dense, AveragePooling2D, GlobalAveragePooling2D, Conv2D
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
    return img[starty:starty+cropy,startx:startx+cropx]

def get_train_generator():
    train_fovea_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        preprocessing_function=crop_center)

    train_context_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.)

    train_fovea_generator = train_fovea_datagen.flow_from_directory(
        os.path.join('data', 'train'),
        target_size=(89, 89),
        batch_size=32,
        classes=data.classes,
        class_mode='categorical')
    
    train_context_generator = train_context_datagen.flow_from_directory(
        os.path.join('data', 'train'),
        target_size=(89, 89),
        batch_size=32,
        classes=data.classes,
        class_mode='categorical',
        interpolation='bilinear')

    while True:
        f = train_fovea_generator.next()
        c = train_context_generator.next()
        yield [f[0],c[0]], f[1]

    # train_generator = zip(train_fovea_generator,train_context_generator)
    # return train_generator, validation_generator

def get_validation_generator():
    context_datagen = ImageDataGenerator(rescale=1./255)
    
    fovea_datagen = ImageDataGenerator(rescale=1./255,
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

def get_model(weights='imagenet'):
    # # create the base pre-trained model
    # base_model = InceptionV3(weights=weights, include_top=False)

    # # add a global spatial average pooling layer
    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # # let's add a fully-connected layer
    # x = Dense(1024, activation='relu')(x)
    # # and a logistic layer
    # predictions = Dense(len(data.classes), activation='softmax')(x)

    # # this is the model we will train
    # model = Model(inputs=base_model.input, outputs=predictions)

    fovea_input = Input(shape=(89,89,3), name='fovea_input')
    x = Conv2D(96,11,3)(fovea_input)
    x = Lambda(local_response_normalization)(x)
    x = AveragePooling2D((2,2),padding='valid')(x)
    x = Conv2D(256,5,1)(x)
    x = Lambda(local_response_normalization)(x)
    x = AveragePooling2D((2,2),padding='valid')(x)
    x = Conv2D(384,1,1)(x)
    x = Conv2D(384,1,1)(x)
    x = Conv2D(256,3,1)(x)

    context_input = Input(shape=(89,89,3), name='context_input')
    c = Conv2D(96,11,3)(context_input)
    c = Lambda(local_response_normalization)(c)
    c = AveragePooling2D((2,2),padding='valid')(c)
    c = Conv2D(256,5,1)(c)
    c = Lambda(local_response_normalization)(c)
    c = AveragePooling2D((2,2),padding='valid')(c)
    c = Conv2D(384,1,1)(c)
    c = Conv2D(384,1,1)(c)
    c = Conv2D(256,3,1)(c)

    cn = concatenate([x,c])

    #cn = AveragePooling2D((2,2),padding='valid')(cn)
    cn = Flatten()(cn)
    fc = Dense(len(data.classes), activation='relu')(cn)
    predictions = Dense(len(data.classes), activation='relu')(fc)

    model = Model(inputs=[fovea_input, context_input], outputs=predictions)    
    #model([fovea_input, context_input])
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9) ,loss='categorical_crossentropy', metrics=['accuracy','top_k_categorical_accuracy'], run_eagerly=True)
    
    return model

def freeze_all_but_top(model):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:-2]:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def freeze_all_but_mid_and_top(model):
    """After we fine-tune the dense layers, train deeper."""
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model

def train_model(model, nb_epoch, train_generator, callbacks=[], validation_generator=None):
    #train_generator, validation_generator = generators
    model.fit(
        train_generator,
        steps_per_epoch=10,
        validation_data=validation_generator,
        validation_steps=10,
        epochs=nb_epoch,
        callbacks=callbacks)
    return model

def main(weights_file):
    model = get_model()
    print(model.summary())
    plot_model(model, to_file='model.png',show_shapes=True)
    

    train_generator = get_train_generator()
    validation_generator = get_validation_generator()
    #generators = (train_generator,validation_generator)

    # if weights_file is None:
    #     print("Loading network from ImageNet weights.")
    #     # Get and train the top layers.
    #     model = freeze_all_but_top(model)
    #     model = train_model(model, 10, generators)
    # else:
    #     print("Loading saved model: %s." % weights_file)
    #     model.load_weights(weights_file)

    # # Get and train the mid layers.
    # model = freeze_all_but_mid_and_top(model)
    model = train_model(model, 20, train_generator,
                        [checkpointer, tensorboard, csv_logger],
                        validation_generator=validation_generator)

if __name__ == '__main__':
    weights_file = None
    main(weights_file)