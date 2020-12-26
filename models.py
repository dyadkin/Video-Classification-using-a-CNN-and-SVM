"""
A collection of models we'll use to attempt to classify videos.
"""
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Flatten, concatenate, Lambda, Input,\
     Dropout, Dense, MaxPooling2D, MaxPooling3D, Conv2D, Conv3D,\
     ZeroPadding3D, Activation, BatchNormalization
from tensorflow.keras.layers.recurrent import LSTM
from tensorflow.keras.layers.wrappers import TimeDistributed
from tensorflow.keras.regularizers import l2
from collections import deque

import tensorflow as tf
import sys


class ResearchModels():
    def __init__(self, nb_classes, model, model_optimizer, seq_length,
                 saved_model=None, features_length=2048):
        """
        `model` = one of:
            lstm
            lrcn
            mlp
            conv_3d
            c3d
            sf_multires
        `model_optimizer` = one of:
            adam
            sgd
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'lstm':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
        elif model == 'lrcn':
            print("Loading CNN-LSTM model.")
            self.input_shape = (seq_length, 80, 80, 3)
            self.model = self.lrcn()
        elif model == 'mlp':
            print("Loading simple MLP.")
            self.input_shape = (seq_length, features_length)
            self.model = self.mlp()
        elif model == 'conv_3d':
            print("Loading Conv3D")
            self.input_shape = (seq_length, 80, 80, 3)
            self.model = self.conv_3d()
        elif model == 'c3d':
            print("Loading C3D")
            self.input_shape = (seq_length, 80, 80, 3)
            self.model = self.c3d()
        elif model == 'sf_multires':
            fovea_input = Input(shape=(89, 89, 3), name='fovea_input')
            context_input = Input(shape=(89, 89, 3), name='context_input')
            self.model = self.SF_Multires(fovea_input, context_input)
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        if model_optimizer == 'adam':
            optimizer = Adam(lr=1e-5, decay=1e-6)       
        elif model_optimizer == 'sgd':
            optimizer = SGD(lr=0.0001, momentum=0.9)
        else:
            print("Unknown model optimizer.")
            sys.exit()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=metrics)
        print(self.model.summary())

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = Sequential()
        model.add(LSTM(2048, return_sequences=False,
                       input_shape=self.input_shape,
                       dropout=0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def lrcn(self):
        """Build a CNN into RNN.
        Starting version from:
            https://github.com/udacity/self-driving-car/blob/master/
                steering-models/community-models/chauffeur/models.py

        Heavily influenced by VGG-16:
            https://arxiv.org/abs/1409.1556

        Also known as an LRCN:
            https://arxiv.org/pdf/1411.4389.pdf
        """
        def add_default_block(model, kernel_filters, init, reg_lambda):

            # conv
            model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3),
                                             padding='same',
                                             kernel_initializer=init,
                                             kernel_regularizer=l2(reg_lambda)
                                             )
                                      )
                      )
            model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Activation('relu')))
            # conv
            model.add(TimeDistributed(Conv2D(kernel_filters, (3, 3),
                                             padding='same',
                                             kernel_initializer=init,
                                             kernel_regularizer=l2(reg_lambda)
                                             )
                                      )
                      )
            model.add(TimeDistributed(BatchNormalization()))
            model.add(TimeDistributed(Activation('relu')))
            # max pool
            model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

            return model

        initialiser = 'glorot_uniform'
        reg_lambda = 0.001

        model = Sequential()

        # first (non-default) block
        model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),
                                         padding='same',
                                         kernel_initializer=initialiser,
                                         kernel_regularizer=l2(reg_lambda)),
                                  input_shape=self.input_shape))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        model.add(TimeDistributed(Conv2D(32, (3, 3),
                                  kernel_initializer=initialiser,
                                  kernel_regularizer=l2(reg_lambda))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Activation('relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        # 2nd-5th (default) blocks
        model = add_default_block(model, 64,  init=initialiser,
                                  reg_lambda=reg_lambda)
        model = add_default_block(model, 128, init=initialiser,
                                  reg_lambda=reg_lambda)
        model = add_default_block(model, 256, init=initialiser,
                                  reg_lambda=reg_lambda)
        model = add_default_block(model, 512, init=initialiser,
                                  reg_lambda=reg_lambda)

        # LSTM output head
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(256, return_sequences=False, dropout=0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def mlp(self):
        """Build a simple MLP. It uses extracted features as the input
        because of the otherwise too-high dimensionality."""
        # Model.
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def conv_3d(self):
        """
        Build a 3D convolutional network, based loosely on C3D.
            https://arxiv.org/pdf/1412.0767.pdf
        """
        # Model.
        model = Sequential()
        model.add(Conv3D(
            32, (3, 3, 3), activation='relu', input_shape=self.input_shape
        ))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(64, (3, 3, 3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(128, (3, 3, 3), activation='relu'))
        model.add(Conv3D(128, (3, 3, 3), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
        model.add(Conv3D(256, (2, 2, 2), activation='relu'))
        model.add(Conv3D(256, (2, 2, 2), activation='relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Dropout(0.5))
        model.add(Dense(1024))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def c3d(self):
        """
        Build a 3D convolutional network, aka C3D.
            https://arxiv.org/pdf/1412.0767.pdf

        With thanks:
            https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2
        """
        model = Sequential()
        # 1st layer group
        model.add(Conv3D(64, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv1',
                         subsample=(1, 1, 1),
                         input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1'))
        # 2nd layer group
        model.add(Conv3D(128, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv2',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool2'))
        # 3rd layer group
        model.add(Conv3D(256, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv3a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(256, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv3b',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool3'))
        # 4th layer group
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv4a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv4b',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool4'))

        # 5th layer group
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv5a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv5b',
                         subsample=(1, 1, 1)))
        model.add(ZeroPadding3D(padding=(0, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool5'))
        model.add(Flatten())

        # FC layers group
        model.add(Dense(4096, activation='relu', name='fc6'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', name='fc7'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model

    def SF_Multires(self, fovea_input, context_input):
        """
        Build the Single Frame Multiresolution network.
            https://cs.stanford.edu/people/karpathy/deepvideo/deepvideo_cvpr2014.pdf
        """

        # Model
        f = Conv2D(96, 11, 3, padding='same', activation='relu')(fovea_input)
        f = Lambda(tf.nn.local_response_normalization,
                   arguments={'alpha': 1e-4, 'bias': 2})(f)
        f = MaxPooling2D((2, 2), padding='same')(f)
        f = Conv2D(256, 5, 1, activation='relu')(f)
        f = Lambda(tf.nn.local_response_normalization,
                   arguments={'alpha': 1e-4, 'bias': 2})(f)
        f = MaxPooling2D((2, 2), padding='same')(f)
        f = Conv2D(384, 3, 1, padding='same', activation='relu')(f)
        f = Conv2D(384, 3, 1, padding='same', activation='relu')(f)
        f = Conv2D(256, 3, 1, padding='same', activation='relu')(f)

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

        cn = concatenate([f, c])

        cn = Flatten()(cn)
        fc = Dense(4096, activation='relu')(cn)
        do = Dropout(0.5)(fc)
        fc = Dense(4096, activation='relu')(fc)
        do = Dropout(0.5)(fc)

        predictions = Dense(len(self.nb_classes), activation='softmax')(do)

        model = Model(inputs=[fovea_input, context_input], outputs=predictions)
        return model
