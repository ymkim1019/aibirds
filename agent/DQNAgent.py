import numpy as np
from PIL import Image
from EventTask import EventTask
from Observation import Observation
from Configuration import globalConfig
from Agent import Agent
#import tensorflow as tf
import math
from collections import deque
import random

class DQNAgent(Agent):
    # From the environments to the agent
    OBSERVE = 0

    # From the agent to the environments
    ACT = 0

    def __init__(self):
        super(DQNAgent, self).__init__()

        self.OBSERVE_SIZE = globalConfig.OBSERVE_SIZE
        self.ANGLE_MAX = 70
        self.ANGLE_MIN = 20
        self.ANGLE_NUM = 10
        self.ANGLE_STEP = (self.ANGLE_MAX - self.ANGLE_MIN) / self.ANGLE_NUM
        self.MEMORY_SIZE = 1000000
        self.PATH = 'observations/'

        self.observe_num = self.__get_observe_num()
        print(self.observe_num)

        self.replayMemory = deque()
        self.init_replayMemory()

    def init_replayMemory(self):
        prev_memory = None
        # memory: state, action, reward, next_state, terminal
        # info: action, reward, terminal, birds_seq, etc..
        for i in range(self.observe_num):
            img, info = self.ob_load(i)
            if prev_memory != None:
                if prev_memory[4] == 0: # non-terminal
                    prev_memory[3] = (img, info[3:])
            self.replayMemory.append([(img, info[3:]), info[0], info[1], (img, info[3:]), info[2]])
            prev_memory = self.replayMemory[-1]

    def __get_observe_num(self):
        import os
        filenames = os.listdir(self.PATH+'img')
        return len(filenames)

    def ob_save(self, img, info):
        np.save(self.PATH+'img/'+str(self.observe_num)+'.npy', img)
        np.save(self.PATH+'info/'+str(self.observe_num)+'.npy', info)
        # info: action, reward, terminal, birds_seq, pigs_num
        self.observe_num += 1

    def ob_load(self, observe_num):
        img = np.load(self.PATH+'img/'+str(observe_num)+'.npy')
        info = np.load(self.PATH+'info/'+str(observe_num)+'.npy')
        return img, info

    def do_job(self, job_obj):
        (job_id, data, env_proxy) = job_obj
        if self.verbose:
            print(str.format("Processing Job id:{}..", job_id))

        if job_id == self.OBSERVE:
            ob = Observation(data)
            self.ob_save(ob)



            self.replayMemory.append((ob.img, action, reward, newState, terminal))
            # print(ob.birds_seq)


            # decision notification
            env_proxy.execute(30) # temp implementation


'''
class Observation:
    def __init__(self):
        self.birds_seq = np.array([4, 4, 4, 4, 0, 0, 0, 0, 0, 0])
        im = Image.open('sample.jpg')
        rgbpix = np.array(im)
        hsvpix = np.array(im.convert('HSV'))
        img = np.zeros((globalConfig.OBSERVE_SIZE, globalConfig.OBSERVE_SIZE))
        for w in range(globalConfig.OBSERVE_SIZE):
            for h in range(globalConfig.OBSERVE_SIZE):
                r, g, b = rgbpix[h][w]
                _, s, v = hsvpix[h][w]

                if (s > 50 and v < 50 and g > 20 and r > 30):  # hill
                    img[h][w] = globalConfig.TYPE_DICT['Hill']
                elif (v > 100 and s < 50):  # stone
                    img[h][w] = globalConfig.TYPE_DICT['Stone']
                elif (r > 150 and g > 100):  # wood
                    img[h][w] = globalConfig.TYPE_DICT['Wood']
                elif (b > 200):  # ice
                    img[h][w] = globalConfig.TYPE_DICT['Ice']
                elif (g > 200):  # pig
                    img[h][w] = globalConfig.TYPE_DICT['Pig']
        self.img = img
        self.pigs_num = 1
        self.prev_stars = 0
        self.first_shot = 1

    def get_state(self):
        temp = np.array([self.pigs_num, self.prev_stars, self.first_shot])
        info = np.append(self.birds_seq, temp)
        return self.img, info


def main():
    agent = DQNAgent()
    ob = Observation()
    agent.save(ob)
    img, info = agent.load(0)
    print(info)




main()'''