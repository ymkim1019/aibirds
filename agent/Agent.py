import numpy as np
import struct
import io
from PIL import Image
from EventTask import EventTask
import queue
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

            imagedata = data[:-16]
            fake_file = io.BytesIO()
            fake_file.write(imagedata)
            im = Image.open(fake_file)
            im_arr = np.array(im)
            #im.show()
            im.save("birds.jpg")

            sling_x = int.from_bytes(data[-16:-12], byteorder='big')
            sling_y = int.from_bytes(data[-12:-8], byteorder='big')
            sling_width = int.from_bytes(data[-8:-4], byteorder='big')
            sling_height = int.from_bytes(data[-4:], byteorder='big')
            #struct.unpack('>I',splingdata)
            print(sling_x,sling_y,sling_width,sling_height)
            # do something..
            im = im.crop((240,180,840,480))
            #state.show()
            im = im.resize((300,150),Image.ANTIALIAS)
            #state.show()
            pix = np.array(im)
            state = np.zeros((150,300))
            for w in range(300):
              for h in range(150):
                pixwh = pix[h][w]
                if(np.linalg.norm(pixwh-[52,35,19]) <5):#hill
                  state[h,w] = 1
                if(np.linalg.norm(pixwh-[160,160,160]) <5):#stone
                  state[h,w] = 2
                if(np.linalg.norm(pixwh-[227,143,29]) <5):#wood
                  state[h,w] = 3
                if(np.linalg.norm(pixwh-[112,203,247]) <5):#ice
                  state[h,w] = 4
                if(np.linalg.norm(pixwh-[95,224,71]) < 5):#pig
                  state[h,w] = 5
            com_thread.queue.put(state)

            # decision notification
            com_thread.send_data(self.FROM_AGENT_TO_ENV_DO_ACTION, struct.pack("iii", -400,400,0)) # temp implementation