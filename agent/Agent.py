import numpy as np
import struct
import io
from PIL import Image
from EventTask import EventTask

class Agent(EventTask):
    # From the environments to the agent
    FROM_ENV_TO_AGENT_REQUEST_FOR_ACTION = 0

    # From the agent to the environments
    FROM_AGENT_TO_ENV_DO_ACTION = 0

    def __init__(self):
        super(Agent, self).__init__('Agent')

    def do_job(self, job_obj):
        (job_id, data, com_thread) = job_obj
        if self.verbose:
            print(str.format("Processing Job id:{}..", job_id))

        if job_id == self.FROM_ENV_TO_AGENT_REQUEST_FOR_ACTION:
            # 2017-06-03 : jyham -- get birds sequence
            max_birds_num = 10
            birds_seq = []
            for i in range(max_birds_num):
                print(struct.unpack("!i", data[i * 4:(i + 1) * 4])[0])
                birds_seq.append(struct.unpack("!i", data[i * 4:(i + 1) * 4])[0])
            print(birds_seq)

            fake_file = io.BytesIO()
            fake_file.write(data[4*max_birds_num:])
            im = Image.open(fake_file)
            im_arr = np.array(im)
            im.show()

            # do something..

            # decision notification
            com_thread.send_data(self.FROM_AGENT_TO_ENV_DO_ACTION, struct.pack("i", 1234)) # temp implementation
