import numpy as np
from PIL import Image
from EventTask import EventTask
#from Observation import Observation
from Configuration import globalConfig
from Agent import Agent
import tensorflow as tf
import math
from collections import deque
import random

class DQNetwork():

    def __init__(self, train, load_step):

        # fixed random seed
        np.random.seed(1)
        tf.set_random_seed(1)

        self.SAVE_PATH = 'dqn_networks/'

        self.OBSERVE_SIZE = globalConfig.OBSERVE_SIZE
        self.ANGLE_MAX = 70
        self.ANGLE_MIN = 20
        self.ANGLE_NUM = 10
        self.ANGLE_STEP = (self.ANGLE_MAX - self.ANGLE_MIN) / self.ANGLE_NUM

        # init experiance replay memory
        self.replayMemory = deque()

        self.GAMMA = 0.9
        self.epsilon = 1.0
        self.EPSILON_DECAY = 0.9
        self.EPSILON_MIN = 0.01

        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 32
        self.time_step = 0

        self.P_KEEP_HIDDEN = 0.8
        self.UPDATE_TIME = 1000

        # init Q network
        self.input_image, self.Q_value = self.createQNetwork()

        # save and load
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        if train:
            self.trainQNetwork()

        #self.createTrainingMethod()

    def getAction(self):
        Q_value = self.Q_value.eval(feed_dict={self.input_image:[self.currentState]})[0]
        action = np.zeros(self.actions)
        action_index = 0
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.ANGLE_NUM)
            action[action_index] = 1
        else:
            action_index = np.argmax(Q_value)
            action[action_index] = 1


    def createTrainingMethod(self):
        self.action_input = tf.placeholder("float", [None, self.ACTION_NUM])  # one-hot. max?
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.mul(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(self.cost) # change optimizer?

    def trainQNetwork(self):
        # 1: Obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, self.BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # 2: calculate y
        y_batch = []
        Q_value_batch = self.t_Q_value.eval(feed_dict={self.t_input_image:nextState_batch})
        for i in range(self.BATCH_SIZE):
            terminal = minibatch[i][4] # terminal? why?
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.GAMMA * np.max(Q_value_batch[i]))

        self.train_op.run(feed_dict = {
            self.y_input : y_batch,
            self.action_input : action_batch,
            self.input_image : state_batch
        })

        if self.time_step % 1000 == 0:
            save_path = self.saver.save(self.session, self.SAVE_PATH, global_step = self.time_step)
            print ("Model saved in file: "+save_path)



    def createQNetwork(self):
        # input layer
        input_image = tf.placeholder(tf.float32, [None, self.OBSERVE_SIZE, self.OBSERVE_SIZE])
        input_layer = tf.reshape(input_image, [-1, self.OBSERVE_SIZE, self.OBSERVE_SIZE, 1])
        # shape = (?, 84, 84, 1)

        # [height, width, input channel, output channel]
        W_conv1 = self.weight_variable([8,8,1,32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(input_layer,W_conv1,2)+b_conv1) # change stride? shape = (?, 42, 42, 32)
        h_pool1 = self.max_pool_2x2(h_conv1) # shape = (?, 21, 21, 32)
        h_pool1 = tf.nn.dropout(h_pool1, self.P_KEEP_HIDDEN)

        W_conv2 = self.weight_variable([4,4,32,64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 1)+b_conv2) # shape = (?, 21, 21, 64)
        h_pool2 = self.max_pool_2x2(h_conv2) # shape = (?, 11, 11, 64)
        h_pool2 = tf.nn.dropout(h_pool2, self.P_KEEP_HIDDEN)

        W_conv3 = self.weight_variable([3,3,64,64])
        b_conv3 = self.bias_variable([64])
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3, 1)+b_conv3) # shape = (?, 11, 11, 64)
        h_pool3 = self.max_pool_2x2(h_conv3) # shape = (?, 6, 6, 64)
        h_pool3 = tf.nn.dropout(h_pool3, self.P_KEEP_HIDDEN)
        h_pool3_shape = h_pool3.get_shape().as_list()
        dim = h_pool3_shape[1]*h_pool3_shape[2]*h_pool3_shape[3]
        h_pool3_flat = tf.reshape(h_pool3, [-1, dim])  # reshape to (?, 2304)

        W_fc1 = self.weight_variable([dim, 512])
        b_fc1 = self.bias_variable([512])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1)+b_fc1)
        h_fc1 = tf.nn.dropout(h_fc1, self.P_KEEP_HIDDEN)

        W_fc2 = self.weight_variable([512, self.ACTION_NUM])
        b_fc2 = self.weight_variable([self.ACTION_NUM])

        # Q value layer
        Q_value = tf.matmul(h_fc1, W_fc2)+b_fc2
        return input_image, Q_value

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        # [batch, height, width, channels]
        return tf.nn.conv2d(x,W,strides=[1, stride, stride,1], padding="SAME")
        # padding = "VALID" or "SAME"

    def max_pool_2x2(self,x):
        # [batch, height, width, channels]
        return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME")

    def get_angle(self, action):
        return self.ANGLE_STEP * action



    def do_job(self, job_obj):
        (job_id, data, env_proxy) = job_obj
        if self.verbose:
            print(str.format("Processing Job id:{}..", job_id))

        if job_id == self.OBSERVE:
            #ob = Observation(data)
            # print(ob.birds_seq)


            # decision notification
            env_proxy.execute(30) # temp implementation


## Temp
class Observation:
    def __init__(self):
        self.birds_seq = [4,4,4,4,0,0,0,0,0,0]
        im = Image.open('sample.jpg')
        rgbpix = np.array(im)
        hsvpix = np.array(im.convert('HSV'))
        self.state = np.zeros((globalConfig.OBSERVE_SIZE, globalConfig.OBSERVE_SIZE))
        for w in range(globalConfig.OBSERVE_SIZE):
            for h in range(globalConfig.OBSERVE_SIZE):
                r,g,b = rgbpix[h][w]
                _,s,v = hsvpix[h][w]

                if (s > 50 and v < 50 and g > 20 and r > 30):  # hill
                    self.state[h][w] = globalConfig.TYPE_DICT['Hill']
                elif (v > 100 and s < 50):  # stone
                    self.state[h][w] = globalConfig.TYPE_DICT['Stone']
                elif (r > 150 and g > 100):  # wood
                    self.state[h][w] = globalConfig.TYPE_DICT['Wood']
                elif (b > 200):  # ice
                    self.state[h][w] = globalConfig.TYPE_DICT['Ice']
                elif (g > 200):  # pig
                    self.state[h][w] = globalConfig.TYPE_DICT['Pig']

        self.pigs_num = 1
        self.prev_stars = 0
        self.first_shot = 1
        self.terminal = 1

def main():
    agent = DQNAgent()
    ob = Observation()
    '''
    f = open("test.txt","w")

    for w in range(84):
        for h in range(84):
            if ob.state[w][h] == 0:
                f.write(' ')
            elif ob.state[w][h] == globalConfig.TYPE_DICT['Hill']:
                f.write('H')
            elif ob.state[w][h] == globalConfig.TYPE_DICT['Stone']:
                f.write('S')
            elif ob.state[w][h] == globalConfig.TYPE_DICT['Wood']:
                f.write('W')
            elif ob.state[w][h] == globalConfig.TYPE_DICT['Ice']:
                f.write('I')
            elif ob.state[w][h] == globalConfig.TYPE_DICT['Pig']:
                f.write('P')
        f.write('\n')

    f.close()
    '''



main()