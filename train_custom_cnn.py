"""
Train on images split into directories. This assumes we've split
our videos into frames and moved them to their respective folders.

Based on:
https://keras.io/preprocessing/image/
and
https://keras.io/applications/
"""
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, concatenate, Lambda, Input,\
     Dropout, Dense, MaxPooling2D, Conv2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard,\
     EarlyStopping, CSVLogger, LearningRateScheduler
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from data import DataSet
import time
import os.path

data = DataSet()


def scheduler(epoch, lr):
    if epoch % 3 == 0:
        return lr * tf.math.exp(-0.5)
    else:
        return lr


# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath=os.path.join('data', 'checkpoints',
                          'SF_MultiRes.{epoch:03d}-{val_accuracy:.2f}.hdf5'),
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir=os.path.join('data', 'logs'))

# Helper: Save results.
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('data', 'logs', 'SF_MultiRes' + '-' +
                                    'training-' + str(timestamp) + '.log'))


# Helper: Schedule learning rate.
lr_scheduler = LearningRateScheduler(scheduler)


def crop_center(img):
    cropx = 89
    cropy = 89
    y, x, z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]-96


def subtract_mean(img):
    return img-96


def get_train_generator():
    train_fovea_datagen = ImageDataGenerator(
        # rescale=1./255,
        # shear_range=0.2,
        horizontal_flip=True,
        rotation_range=20.,
        preprocessing_function=crop_center)

    train_context_datagen = ImageDataGenerator(
        # rescale=1./255,
        # shear_range=0.2,
        horizontal_flip=True,
        rotation_range=20.,
        preprocessing_function=subtract_mean)

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
        yield [f[0], c[0]], f[1]

    # train_generator = zip(train_fovea_generator,train_context_generator)
    # return train_generator, validation_generator


def get_validation_generator():
    context_datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20.,
        preprocessing_function=subtract_mean)  # rescale=1./255)

    fovea_datagen = ImageDataGenerator(
        # rescale=1./255,
        horizontal_flip=True,
        rotation_range=20.,
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
        yield [f[0], c[0]], f[1]


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

    fovea_input = Input(shape=(89, 89, 3), name='fovea_input')
    x = Conv2D(96, 11, 3, padding='same', activation='relu')(fovea_input)
    x = Lambda(tf.nn.local_response_normalization,
               arguments={'alpha': 1e-4, 'bias': 2})(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(256, 5, 1, activation='relu')(x)
    x = Lambda(tf.nn.local_response_normalization,
               arguments={'alpha': 1e-4, 'bias': 2})(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(384, 3, 1, padding='same', activation='relu')(x)
    x = Conv2D(384, 3, 1, padding='same', activation='relu')(x)
    x = Conv2D(256, 3, 1, padding='same', activation='relu')(x)

    context_input = Input(shape=(89, 89, 3), name='context_input')
    c = Conv2D(96, 11, 3, padding='same', activation='relu')(context_input)
    c = Lambda(tf.nn.local_response_normalization,
               arguments={'alpha': 1e-4, 'bias': 2})(c)
    c = MaxPooling2D((2, 2), padding='same')(c)
    c = Conv2D(256, 5, 1, activation='relu')(c)
    c = Lambda(tf.nn.local_response_normalization,
               arguments={'alpha': 1e-4, 'bias': 2})(c)
    c = MaxPooling2D((2, 2), padding='same')(c)
    c = Conv2D(384, 3, 1, padding='same', activation='relu')(c)
    c = Conv2D(384, 3, 1, padding='same', activation='relu')(c)
    c = Conv2D(256, 3, 1, padding='same', activation='relu')(c)

    cn = concatenate([x, c])

    cn = Flatten()(cn)
    fc = Dense(4096, activation='relu')(cn)
    do = Dropout(0.5)(fc)
    fc = Dense(4096, activation='relu')(fc)
    do = Dropout(0.5)(fc)

    predictions = Dense(len(data.classes), activation='softmax')(do)

    model = Model(inputs=[fovea_input, context_input], outputs=predictions)
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'top_k_categorical_accuracy'])  # , run_eagerly=False)
    return model


def freeze_all_but_top(model):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:-2]:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to
    # non-trainable)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

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


def train_model(model, nb_epoch, train_generator, callbacks=[],
                validation_generator=None):
    # train_generator, validation_generator = generators
    model.fit(
        train_generator,
        steps_per_epoch=1000,
        validation_data=validation_generator,
        validation_steps=10,
        epochs=nb_epoch,
        callbacks=callbacks)
    return model


def main(weights_file):
    model = get_model()
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)

    train_generator = get_train_generator()
    validation_generator = get_validation_generator()
    # generators = (train_generator,validation_generator)

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
    model = train_model(model, 1250, train_generator,
                        [checkpointer, tensorboard, csv_logger, lr_scheduler],
                        validation_generator=validation_generator)


if __name__ == '__main__':
    weights_file = None
    main(weights_file)
