import numpy as np
from Configuration import globalConfig
import tensorflow as tf

class RetraceNetwork():
    def __init__(self, train, load_step):
        # fix random seed
        np.random.seed(1)
        tf.set_random_seed(1)

        self.SAVE_PATH = 'retrace/weight/'

        self.OBSERVE_SIZE = globalConfig.OBSERVE_SIZE
        self.ANGLE_MAX = globalConfig.ANGLE_MAX
        self.ANGLE_MIN = globalConfig.ANGLE_MIN
        self.ANGLE_NUM = globalConfig.ANGLE_NUM
        self.ANGLE_STEP = globalConfig.ANGLE_STEP

        self.GAMMA = 0.9
        self.epsilon = 1.0
        self.EPSILON_DECAY = 0.9
        self.EPSILON_MIN = 0.01

        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 32
        self.time_step = 0

        self.P_KEEP_HIDDEN = 0.8
        self.UPDATE_TIME = 1000

    def createNetwork(self):
        # input layer
        input_image = tf.placeholder(tf.float32, [None, self.OBSERVE_SIZE, self.OBSERVE_SIZE, 1]) # (?, 84, 84, 1)
        input_info = tf.placeholder(tf.float32, [None, globalConfig.MAX_BIRDS_SEQ]) # (?, 10)
        input_action = tf.placeholder(tf.float32, [None, self.ANGLE_NUM]) # (?, action num)

        # [height, width, input channel, output channel]
        W_conv1 = self.weight_variable([8, 8, 1, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(input_image, W_conv1, 2) + b_conv1)  # change stride? shape = (?, 42, 42, 32)
        h_pool1 = self.max_pool_2x2(h_conv1)  # shape = (?, 21, 21, 32)
        h_pool1 = tf.nn.dropout(h_pool1, self.P_KEEP_HIDDEN)

        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 1) + b_conv2)  # shape = (?, 21, 21, 64)
        h_pool2 = self.max_pool_2x2(h_conv2)  # shape = (?, 11, 11, 64)
        h_pool2 = tf.nn.dropout(h_pool2, self.P_KEEP_HIDDEN)

        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3, 1) + b_conv3)  # shape = (?, 11, 11, 64)
        h_pool3 = self.max_pool_2x2(h_conv3)  # shape = (?, 6, 6, 64)
        h_pool3 = tf.nn.dropout(h_pool3, self.P_KEEP_HIDDEN)
        h_pool3_shape = h_pool3.get_shape().as_list()
        dim = h_pool3_shape[1] * h_pool3_shape[2] * h_pool3_shape[3]
        h_pool3_flat = tf.reshape(h_pool3, [-1, dim])  # reshape to (?, 2304)




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

    def angle2num(self, angle):
        return int((angle - self.ANGLE_MIN)/self.ANGLE_STEP)

rn = RetraceNetwork(1, 1)
print(rn.angle2num(20))