import struct
import io
from PIL import Image
import numpy as np
from Configuration import globalConfig


class Observe:
    def __init__(self, data):
        self.birds_seq = self.__get_birds_seq(data[:4*globalConfig.MAX_BIRDS_NUM])
        self.im = self.__get_image(data[4*globalConfig.MAX_BIRDS_NUM:])

    def __get_birds_seq(self, data):
        birds_seq = []
        for i in range(globalConfig.MAX_BIRDS_NUM):
            birds_seq.append(struct.unpack("!i", data[i*4:(i+1)*4])[0])
        return birds_seq

    def __get_image(self, data):
        fake_file = io.BytesIO()
        fake_file.write(data)
        im = Image.open(fake_file)
        if globalConfig.SHOW:
            im.show()
        im_arr = np.array(im.convert('L'))
        im = im[:,:,np.newaxis]
        return im