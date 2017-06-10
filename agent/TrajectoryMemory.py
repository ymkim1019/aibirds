from collections import deque
import random
import numpy as np
import json
from Configuration import globalConfig
#import pickle

class TrajectoryMemory:
    def __init__(self):
        self.size = 0
        self.total_size = 0
        self.memory = deque()
        self.MAX_SIZE = 1000 # maximum number of episodes
        self.PATH = 'retrace/memory_12_001/'
        self.init_load()
        np.random.seed(1)

    def add(self, img, info, action, reward, fst_shot, policy):
        #print("seq: " + str(info) + " act: " + str(action)+" reward: "+str(reward)+" fst: "+str(fst_shot)+" policy: "+str(policy))
        if self.size > self.MAX_SIZE:
            self.memory.leftpop()
            self.size -= 1
        if action < globalConfig.ANGLE_MIN or action > globalConfig.ANGLE_MAX:
            print ("improper action : "+action)
            return
        data = [img, info, action, reward, policy]

        if fst_shot:
            print('fst_shot')
            if self.size > 0:
                self.save(self.total_size-1, self.memory[-1])
            #print (data)
            #print (self.memory)
            self.memory.append([data])
            self.size += 1
            self.total_size += 1
        else:
            self.memory[-1].append(data)

    def save(self, ep_num, episode):
        fname = self.PATH+'episode-'+str(ep_num)+'.json'
        jsonstring = json.dumps(episode)

        f = open(fname, 'w')
        f.write(jsonstring)
        #pickle.dump(episode, f)
        f.close()
        print("episode saved into "+fname)

    def init_load(self):
        import os
        self.total_size = len(os.listdir(self.PATH))
        start = 0
        if self.total_size > self.MAX_SIZE:
            start = self.total_size - self.MAX_SIZE
        for i in range(start, self.total_size):
            print(i)
            fname = self.PATH+'episode-'+str(i)+'.json'
            f = open(fname, 'r')
            data = json.loads(f.readline())
            #data = pickle.load(f)
            f.close
            self.memory.append(data)
            self.size += 1

    def batch(self, batch_size):
        if batch_size > self.size:
            sampled = np.arange(self.size)
            np.random.shuffle(sampled)
        else:
            sampled = np.random.randint(0, self.size, batch_size)
        return [self.memory[i] for i in sampled]
