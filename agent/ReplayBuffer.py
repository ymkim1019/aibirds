from collections import deque
import random
import pickle
from Configuration import globalConfig

class ReplayBuffer(object):
    PICKLE_PATH = 'replay.dat'

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0

        try:
            f = open(self.PICKLE_PATH, 'rb')
            self.buffer = pickle.load(f)
            self.num_experiences = len(self.buffer)
            print('replay buffer loaded')
        except Exception as e:
            print(e)
            self.buffer = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add_dqn_exp(self, state, action, reward, new_state, new_actions, done):
        experience = (state, action, reward, new_state, new_actions, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

        if self.count() % globalConfig.replay_buf_dump_interval == 0:
            with open(self.PICKLE_PATH, 'wb') as f:
                pickle.dump(self.buffer, f)
                print("dump replay buffer")

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0
