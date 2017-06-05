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

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class CriticNetwork(object):
    def __init__(self, sess, TAU, LEARNING_RATE):
        self.sess = sess
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network()
        self.target_model, self.target_action, self.target_state = self.create_critic_network()
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self):
        print("Now we build the model")

        # inputs
        input_2d = Input(shape=(globalConfig.FRAME_WIDTH, globalConfig.FRAME_HEIGHT, 1)) # 2d
        input_num_objects = Input(shape=(5,)) # num of pigs, stones, woods, ices, tnts
        input_bird = Input(shape=(1,)) # bird on the sling
        A = Input(shape=(3,)) # target, high_shot, tap

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
        fc_input = concatenate([conv_out, input_num_objects, input_bird, A], axis=-1)
        x = Dense(512, activation='relu')(fc_input)
        x = Dense(512, activation='relu')(x)
        V = Dense(1, activation='linear')(x)

        S = [input_2d, input_num_objects, input_bird]
        model_input = S + [A]
        model = Model(inputs=model_input, outputs=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    critic = CriticNetwork(sess, globalConfig.TAU, globalConfig.LRA)
    critic.create_critic_network()

    pixels = np.array(np.random.rand(globalConfig.FRAME_WIDTH, globalConfig.FRAME_HEIGHT, 1))
    num_objects = np.array([3, 2, 3, 4, 0])
    input_bird = np.array([1])
    a_t = np.array([0.2, 0.3, 0.5])

    pixels = pixels.reshape(tuple([1]) + pixels.shape)
    num_objects = num_objects.reshape(tuple([1]) + num_objects.shape)
    input_bird = input_bird.reshape(tuple([1]) + input_bird.shape)
    a_t = a_t.reshape(tuple([1]) + a_t.shape)

    print(pixels.shape, num_objects.shape, input_bird.shape, a_t.shape)

    q_s_a = critic.model.predict([pixels, num_objects, input_bird, a_t])

    print(q_s_a)
