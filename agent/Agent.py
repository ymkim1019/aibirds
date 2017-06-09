import numpy as np
import struct
import io
from PIL import Image
from EventTask import EventTask
import queue
import math
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

            imagedata = data[:-20]
            fake_file = io.BytesIO()
            fake_file.write(imagedata)
            im = Image.open(fake_file)
            im_arr = np.array(im)
            #im.show()
            #im.save("birds.jpg")

            sling_x = int.from_bytes(data[-20:-16], byteorder='big')
            sling_y = int.from_bytes(data[-16:-12], byteorder='big')
            sling_width = int.from_bytes(data[-12:-8], byteorder='big')
            sling_height = int.from_bytes(data[-8:-4], byteorder='big')
            numpigs = int.from_bytes(data[-4:], byteorder='big')
            #struct.unpack('>I',splingdata)
            print(sling_x,sling_y,sling_width,sling_height,numpigs)
            # do something..
            com_thread.state_queue.put([im,sling_x,sling_y,sling_width,sling_height])

            action_angle = com_thread.action_queue.get()
            r = 600
            dx = round(- r * math.cos(action_angle * math.pi / 180))
            dy = round(r * math.sin(action_angle * math.pi / 180))
            # decision notification
            com_thread.send_data(self.FROM_AGENT_TO_ENV_DO_ACTION, struct.pack("ii", dx,dy)) # temp implementation