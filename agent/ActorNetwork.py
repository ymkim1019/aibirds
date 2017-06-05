import numpy as np
import math
from keras.initializers import RandomNormal
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Conv2D, concatenate
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from Configuration import globalConfig


class ActorNetwork(object):
    def __init__(self, sess, TAU, LEARNING_RATE):
        self.sess = sess
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network()
        self.target_model, self.target_weights, self.target_state = self.create_actor_network()
        self.action_gradient = tf.placeholder(tf.float32, [None, 3])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self):
        print("Now we build the model..actor network")

        # inputs
        input_2d = Input(shape=(globalConfig.FRAME_WIDTH, globalConfig.FRAME_HEIGHT, globalConfig.CHANNELS)) # 2d
        input_num_objects = Input(shape=(5,)) # num of pigs, stones, woods, ices, tnts
        input_bird = Input(shape=(1,)) # bird on the sling

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
        fc_input = concatenate([conv_out, input_num_objects, input_bird], axis=-1)
        x = Dense(512, activation='relu')(fc_input)
        x = Dense(512, activation='relu')(x)

        # target
        target = Dense(1, activation='sigmoid')(x) # floor(target * sum(input_num_objects))

        high_shot = Dense(1, activation='sigmoid')(x) # select high shot if high_shot >= 0.5

        # tap
        tap = Dense(1, activation='sigmoid')(x) # 65 + tap*25 % of the way

        S = [input_2d, input_num_objects, input_bird]
        V = concatenate([target, high_shot, tap], axis=-1)
        model = Model(S, V)

        return model, model.trainable_weights, S


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
    actor = ActorNetwork(sess, globalConfig.TAU, globalConfig.LRA)
    actor.create_actor_network()
    pixels = np.array(np.random.rand(globalConfig.FRAME_WIDTH, globalConfig.FRAME_HEIGHT, 1))
    num_objects = np.array([3, 2, 3, 4, 0])
    input_bird = np.array([1])

    pixels = pixels.reshape(tuple([1]) + pixels.shape)
    num_objects = num_objects.reshape(tuple([1]) + num_objects.shape)
    input_bird = input_bird.reshape(tuple([1]) + input_bird.shape)
    print(pixels.shape, num_objects.shape, input_bird.shape)

    a_t = actor.model.predict([pixels, num_objects, input_bird])

    print(a_t)