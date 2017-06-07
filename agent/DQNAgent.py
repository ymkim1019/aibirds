import numpy as np
from PIL import Image
from EventTask import EventTask
from Observation import Observation
from Configuration import globalConfig
from Agent import Agent
import tensorflow as tf
import math
from collections import deque
import random

class DQNAgent(Agent):
    # From the environments to the agent
    OBSERVE = 0

    # From the agent to the environments
    ACT = 0

    def __init__(self, trainable=1, load_model=1):
        super(DQNAgent, self).__init__()

        np.random.seed(1)

        # init experiance replay memory
        self.replayMemory = deque()

        self.timeStep = 0
        self.epsilon = globalConfig.INITIAL_EPSILON
        self.action_num = globalConfig.ACTION_NUM

        # init Q network


        # Tensorflow GPU optimization
        '''
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.trainable = trainable
        K.set_session(self.sess)

        # actor critic

        self.graph = tf.get_default_graph()
        '''

    def createQNetwork(self):
        # input layer
        inputLayer = tf.placeholder(tf.float32, [None,8,8])

        # hidden Layer




    def do_job(self, job_obj):
        (job_id, data, env_proxy) = job_obj
        if self.verbose:
            print(str.format("Processing Job id:{}..", job_id))

        if job_id == self.OBSERVE:
            ob = Observation(data)
            # print(ob.birds_seq)


            # decision notification
            env_proxy.execute(30) # temp implementation


