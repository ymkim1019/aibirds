import numpy as np
import struct
from EventTask import EventTask


class Agent(EventTask):
    # From the environments to the agent
    FROM_ENV_TO_AGENT_REQUEST_FOR_ACTION = 1

    # From the agent to the environments
    FROM_AGENT_TO_ENV_DO_ACTION = 10

    def __init__(self):
        super(Agent, self).__init__('Agent')

    def do_job(self, job_obj):
        (job_id, data, com_thread) = job_obj
        if self.verbose:
            print(str.format("Processing Job id:{}..", job_id))

        if job_id == self.FROM_ENV_TO_AGENT_REQUEST_FOR_ACTION:
            #env = np.frombuffer(data)
            env = data
            if self.verbose:
                print("env:", env)

            # do something..

            # decision notification
            com_thread.send_data(self.FROM_AGENT_TO_ENV_DO_ACTION, struct.pack("i", 1234)) # temp implementation
