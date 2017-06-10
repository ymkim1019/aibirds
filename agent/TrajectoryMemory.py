from collections import deque
import random
import numpy as np
import json

class TrajectoryMemory:
    def __init__(self):
        self.size = 0
        self.total_size = 0
        self.memory = deque()
        self.MAX_SIZE = 1000 # maximum number of episodes
        self.PATH = 'retrace/temp2/'
        self.init_load()
        np.random.seed(1)

    def add(self, img, info, action, reward, fst_shot, policy):
        if self.size > self.MAX_SIZE:
            self.memory.leftpop()
            self.size -= 1
        data = [img, info, action, reward, policy]
        if fst_shot:
            print('fst_shot')
            if self.size > 0:
                self.save(self.total_size-1, self.memory[-1])
            self.memory.append([data])
            self.size += 1
            self.total_size += 1
        else:
            self.memory[-1].append(data)

    def save(self, ep_num, episode):
        fname = self.PATH+'episode-'+str(ep_num)+'.json'
        f = open(fname,'w')
        f.write(json.dumps(episode))
        print("episode saved into "+fname)
        f.close()

    def init_load(self):
        import os
        self.total_size = len(os.listdir(self.PATH))
        start = 0
        if self.total_size > self.MAX_SIZE:
            start = self.total_size - self.MAX_SIZE
        for i in range(start, self.total_size):
            print(i)
            f = open(self.PATH+'episode-'+str(i)+'.json').read()
            self.memory.append(json.loads(f))
            self.size += 1

    def batch(self, batch_size):
        if batch_size > self.size:
            sampled = np.random.shuffle(np.arange(self.size))
        else:
            sampled = np.random.randint(0, self.size, batch_size)
        return [self.memory[i] for i in sampled]
