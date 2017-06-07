import numpy as np
import struct
import io
from PIL import Image
from EventTask import EventTask
from Observation import Observation
from Configuration import globalConfig

class Agent(EventTask):
    # From the environments to the agent
    OBSERVE = 0

    # From the agent to the environments
    ACT = 0

    def __init__(self):
        super(Agent, self).__init__('Agent')

    def do_job(self, job_obj):
        (job_id, data, env_proxy) = job_obj
        if self.verbose:
            print(str.format("Processing Job id:{}..", job_id))

        if job_id == self.OBSERVE:
            ob = Observation(data)
            #print(ob.birds_seq)

            # decision notification
            env_proxy.execute(30) # temp implementation

