import numpy as np
from PIL import Image
from EventTask import EventTask
from Observation import Observation
from Configuration import globalConfig
from Agent import Agent
import tensorflow as tf
import keras.backend as K
import math

class DDPGAgent(Agent):
    # From the environments to the agent
    OBSERVE = 0

    # From the agent to the environments
    ACT = 0

    def __init__(self, trainable=1, load_model=1):
        super(DDPGAgent, self).__init__()

        np.random.seed(1)

        # Tensorflow GPU optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.trainable = trainable
        K.set_session(self.sess)

        # actor critic

        self.graph = tf.get_default_graph()




    def do_job(self, job_obj):
        (job_id, data, env_proxy) = job_obj
        if self.verbose:
            print(str.format("Processing Job id:{}..", job_id))

        if job_id == self.OBSERVE:
            ob = Observation(data)
            # print(ob.birds_seq)
            if globalConfig.SHOW:
                ob.im.show()

            # decision notification
            env_proxy.execute(30) # temp implementation


