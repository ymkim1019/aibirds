import numpy as np
from Configuration import globalConfig
import tensorflow as tf
from TrajectoryMemory import TrajectoryMemory
class RetraceNetwork():
    def __init__(self, load, traj_mem): # if not load, load_step = 0
        # fix random seed
        np.random.seed(1)
        tf.set_random_seed(1)

        self.SAVE_PATH = 'retrace/weight/'
        self.traj_mem = traj_mem

        self.OBSERVE_SIZE = globalConfig.OBSERVE_SIZE
        self.ANGLE_MAX = globalConfig.ANGLE_MAX
        self.ANGLE_MIN = globalConfig.ANGLE_MIN
        self.ANGLE_NUM = globalConfig.ANGLE_NUM
        self.ANGLE_STEP = globalConfig.ANGLE_STEP
        self.INFO_DIM = globalConfig.MAX_BIRDS_SEQ

        self.GAMMA = 0.9
        self.epsilon = 1.0
        #self.epsilon = 0.5
        self.EPSILON_DECAY = 0.9
        self.EPSILON_MIN = 0.01

        self.LEARNING_RATE = 0.001
        #self.BATCH_SIZE = 32
        self.BATCH_SIZE = 3
        self.epoch = 0

        self.P_KEEP_HIDDEN = 0.8
        self.UPDATE_TIME = 1000

        self.input_image, self.input_info, self.Q_value, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2 = self.createNetwork()
        self.input_imageC, self.input_infoC, self.Q_valueC, W_conv1C, b_conv1C, W_conv2C, b_conv2C, W_conv3C, b_conv3C, W_fc1C, b_fc1C, W_fc2C, b_fc2C = self.createNetwork()
        self.copyOperation = [W_conv1C.assign(W_conv1), b_conv1C.assign(b_conv1), W_conv2C.assign(W_conv2), b_conv2C.assign(b_conv2), W_conv3C.assign(W_conv3),
                              b_conv3C.assign(b_conv3), W_fc1C.assign(W_fc1), b_fc1C.assign(b_fc1), W_fc2C.assign(W_fc2), b_fc2C.assign(b_fc2)]

        self.createTrainMethod()

        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        if load:
            checkpoint = tf.train.get_checkpoint_state(self.SAVE_PATH)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                print ("Model restored from "+checkpoint.model_checkpoint_path)
            else:
                print ("Cannot restore the network")

    def createTrainMethod(self):
        self.input_action = tf.placeholder("float", [None, self.ANGLE_NUM])
        self.input_y = tf.placeholder("float", [None])
        Q_Action = tf.reduce_sum(tf.multiply(self.Q_value, self.input_action), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.input_y - Q_Action))
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.cost)

    def trainNetwork(self):
        minibatch = self.traj_mem.batch(self.BATCH_SIZE)
        img_batch = np.array([np.zeros((84,84)) for episode in minibatch for data in episode])  # (None, 84, 84)
        #img_batch = np.array([data[0] for episode in minibatch for data in episode]) # (None, 84, 84)
        info_batch = np.array([data[1] for episode in minibatch for data in episode]) # (None, 10)
        action_batch = np.array([self.angle2onehot(data[2]) for episode in minibatch for data in episode]) # (None, angle num)
        y_batch = np.array([data for episode in minibatch for data in self.calculate_new_q(episode)]) # (None,)


        _, c = self.session.run([self.optimizer, self.cost], feed_dict = {
            self.input_image: img_batch,
            self.input_info: info_batch,
            self.input_action: action_batch,
            self.input_y: y_batch})
        self.saver.save(self.session, self.SAVE_PATH + 'network', global_step=self.epoch)
        self.copyNetwork()
        self.epoch += 1
        return c


    def calculate_new_q(self, episode):
        #print(episode)
        tot = len(episode)
        #print(tot)
        img_batch = np.array([np.zeros((84,84)) for i in range(tot)])
        #img_batch = np.array([data[0] for data in episode]) # (tot, 84, 84)
        info_batch = np.array([data[1] for data in episode]) # (tot, info_dim)
        action_num_batch = np.array([self.angle2num(data[2]) for data in episode]) # (tot, )
        action_vector_batch = np.array([self.angle2onehot(data[2]) for data in episode]) # (tot, angle_num)
        reward_batch = np.array([data[3] for data in episode]) # (tot, )
        behavior_policy_prob_batch = np.array([data[4] for data in episode]) # (tot, )

        Q_value_batch = self.session.run(self.Q_valueC, feed_dict={self.input_imageC : img_batch, self.input_infoC : info_batch})
        # (tot, angle_num)
        max_action_indices = np.argmax(Q_value_batch, axis=1) # (tot,)
        cur_policy_prob_batch = np.ones((tot,self.ANGLE_NUM))*self.epsilon/self.ANGLE_NUM # (tot, angle_num)
        for i in range(tot):
            cur_policy_prob_batch[i][max_action_indices[i]] += (1-self.epsilon)

        actioned_Q_value_batch = np.sum(np.multiply(Q_value_batch, action_vector_batch), axis=1)  # (tot,)

        if tot == 1:
            td_batch = reward_batch # (tot, )
        else:
            expected_Q_value_batch = np.sum(np.multiply(cur_policy_prob_batch, Q_value_batch), axis=1) # (tot, )

            td_batch = reward_batch + self.GAMMA*np.append(expected_Q_value_batch[1:],0) - np.append(actioned_Q_value_batch[:-2],0)
            # (tot, )


        cur_policy_action_prob_batch = np.sum(np.multiply(cur_policy_prob_batch, action_vector_batch), axis=1) # (tot,)
        cs_batch = np.divide(cur_policy_action_prob_batch, behavior_policy_prob_batch) # (tot,)
        cs_batch = self.GAMMA*np.array([np.minimum(1.0, c) for c in cs_batch]) # (tot,)
        cs_batch = self.multiply_cs(cs_batch) # (tot, tot)
        td_batch = np.tile(td_batch, (tot, 1))
        delta_q_batch = np.sum(np.multiply(cs_batch, td_batch), axis=1) # (tot,)

        return np.add(delta_q_batch, actioned_Q_value_batch)

        #return delta_q_batch


    def multiply_cs(self, cs_batch):
        tot = len(cs_batch)
        res = np.triu(np.ones((tot,tot)),0)
        for i in range(tot):
            for j in range(i,tot):
                res[i][j] *= self.product_cs(i,j,cs_batch)
        return res


    def product_cs(self, start, end, cs_batch):
        if start>end:
            return 0.0
        if start == end:
            return 1.0
        return np.prod(cs_batch[start+1:end+1])


    # def getMaxActionIndex(self, img, info):
    #     Q_value = self.Q_valueC.eval(feed_dict={self.input_imageC : img, self.input_infoC : info})
    #     action_index = np.argmax(Q_value)
    #     return action_index
    #
    # def getActionProb(self, img, info):
    #     max_index = self.getMaxActionIndex(img, info)
    #     prob = np.ones(self.ANGLE_NUM)*self.epsilon/self.ANGLE_NUM
    #     prob[max_index] += (1 - self.epsilon)
    #     return prob


    def copyNetwork(self):
        self.session.run(self.copyOperation)

    def createNetwork(self):
        # input layer
        input_image = tf.placeholder(tf.float32, [None, self.OBSERVE_SIZE, self.OBSERVE_SIZE]) # (?, 84, 84)
        image_reshape = tf.reshape(input_image,[-1,self.OBSERVE_SIZE,self.OBSERVE_SIZE,1]) # (?, 84, 84, 1)
        input_info = tf.placeholder(tf.float32, [None, self.INFO_DIM]) # (?, 10)
        #input_action = tf.placeholder(tf.float32, [None, self.ANGLE_NUM]) # (?, action num)

        # [height, width, input channel, output channel]
        W_conv1 = self.weight_variable([8, 8, 1, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(image_reshape, W_conv1, 2) + b_conv1)  # change stride? shape = (?, 42, 42, 32)
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

        concat_layer = tf.concat([input_info, h_pool3_flat], -1)
        dim = concat_layer.get_shape().as_list()[1]

        W_fc1 = self.weight_variable([dim,512])
        b_fc1 = self.bias_variable([512])
        h_fc1 = tf.nn.relu(tf.matmul(concat_layer, W_fc1)+b_fc1)
        h_fc1 = tf.nn.dropout(h_fc1, self.P_KEEP_HIDDEN)

        W_fc2 = self.weight_variable([512, self.ANGLE_NUM])
        b_fc2 = self.bias_variable([self.ANGLE_NUM])

        Q_value = tf.matmul(h_fc1, W_fc2)+b_fc2
        return input_image, input_info, Q_value, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2


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

    def angle2onehot(self, angle):
        vec = np.zeros(self.ANGLE_NUM)
        vec[self.angle2num(angle)] = 1
        return vec

traj_mem = TrajectoryMemory()
rn = RetraceNetwork(True, traj_mem)
# episode = traj_mem.memory[6]
# for i in range(10):
#     c = rn.trainNetwork()
# print(rn.epoch)
# print(c)

#print (episode[0])
#print(rn.calculate_new_q(episode))
#print(rn.angle2num(20))