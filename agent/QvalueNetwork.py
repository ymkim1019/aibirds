import numpy as np
import math
from keras.initializers import RandomNormal
from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, concatenate, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from Configuration import globalConfig


class QvalueNetwork(object):
    def __init__(self, sess, TAU, LEARNING_RATE):
        self.sess = sess
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model = self.create_network()
        self.target_model = self.create_network()
        self.sess.run(tf.initialize_all_variables())

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.TAU * weights[i] + (1 - self.TAU) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def create_network(self):
        print("Building network....")

        # inputs
        input_2d = Input(shape=(globalConfig.FRAME_WIDTH, globalConfig.FRAME_HEIGHT, 1)) # 2d
        input_num_objects = Input(shape=(5,)) # num of pigs, stones, woods, ices, tnts
        input_bird = Input(shape=(1,)) # bird on the sling
        input_sling_pos = Input(shape=(2,)) # pos of sling
        A = Input(shape=(3,)) # target type, angle, tap

        # CNN
        convModel = Sequential()
        convModel.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'
                                    ,input_shape=(globalConfig.FRAME_WIDTH, globalConfig.FRAME_HEIGHT, 1)))
        convModel.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        convModel.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        convModel.add(Flatten())
        convModel.add(Dense(512, activation='relu'))
        conv_out = convModel(input_2d)

        # FC
        fc_input = concatenate([conv_out, input_num_objects, input_bird, input_sling_pos, A], axis=-1)
        x = Dense(512, activation='relu')(fc_input)
        x = Dense(512, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        V = Dense(1, activation='linear')(x)

        S = [input_2d, input_num_objects, input_bird, input_sling_pos]
        model_input = S + [A]
        model = Model(inputs=model_input, outputs=V)
        adam = Adam(lr=self.LEARNING_RATE, clipvalue=1)
        model.compile(loss='mse', optimizer=adam)
        return model