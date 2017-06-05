import numpy as np
import struct
import io
from PIL import Image
from EventTask import EventTask
from Configuration import globalConfig
from Observe import Observe

class Agent(EventTask):
    # ENV -> AGENT
    OBSERVE = 0

    # AGENT -> ENV
    ACT = 0

    def __init__(self):
        super(Agent, self).__init__('Agent')

    def do_job(self, job_obj):
        (job_id, ob, env_proxy) = job_obj
        if self.verbose:
            print(str.format("Processing Job id:{}..", job_id))

        if job_id == self.OBSERVE:
            print(ob.birds_seq)
            print(ob.im.shape)

            # do something..

            # execute action
            env_proxy.send_data(self.ACT, 10)
