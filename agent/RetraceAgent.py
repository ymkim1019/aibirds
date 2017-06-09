import numpy as np
import struct
import io
from PIL import Image
from EventTask import EventTask
from Observation import Observation
from Configuration import globalConfig
from Agent import Agent
from TrajectoryMemory import TrajectoryMemory

class RetraceAgent(Agent):
    # From the environments to the agent
    OBSERVE = 0

    # From the agent to the environments
    ACT = 0

    def __init__(self):
        super(RetraceAgent, self).__init__()

        self.traj_mem = TrajectoryMemory()
        self.prev_img = None
        self.prev_info = None
        self.prev_action = None
        self.prev_first = True

    def do_job(self, job_obj):
        (job_id, data, env_proxy) = job_obj
        if self.verbose:
            print(str.format("Processing Job id:{}..", job_id))

        if job_id == self.OBSERVE:
            ob = Observation(data)
            print(ob.birds_seq)
            action = 30*(np.random.randint(2)+1)

            prev_reward = - ob.pigs_num
            if ob.first_shot:
                prev_reward = ob.prev_stars

            if not self.prev_img == None:
                self.traj_mem.add(self.prev_img, self.prev_info, self.prev_action, prev_reward, self.prev_first)

            print(self.traj_mem.memory)
            self.prev_img = ob.img
            self.prev_info = ob.birds_seq
            self.prev_action = action
            self.prev_first = ob.first_shot
            print('give action '+str(action))
            # decision notification
            env_proxy.execute(action) # temp implementation

