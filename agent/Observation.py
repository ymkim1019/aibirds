import struct
import io
from PIL import Image
import numpy as np
from Configuration import globalConfig

class Observation:
    def __init__(self, data):
        split = [4*globalConfig.MAX_BIRDS_SEQ, 4*globalConfig.MAX_BIRDS_SEQ+16]
        self.birds_seq = self.__get_birds_seq(data[:split[0]])
        sling_x, sling_y, sling_height, ground = self.__get_sling(data[split[0]:split[1]])
        self.im = self.__get_image(data[split[1]:], sling_x, sling_y, sling_height, ground)

    def __get_birds_seq(self, data):
        birds_seq = []
        for i in range(globalConfig.MAX_BIRDS_SEQ):
            birds_seq.append(struct.unpack("!i", data[i * 4:(i + 1) * 4])[0])
        return birds_seq

    def __get_sling(self, data):
        x = struct.unpack("!i", data[:4])[0]
        y = struct.unpack("!i", data[4:8])[0]
        height = struct.unpack("!i", data[8:12])[0]
        ground = struct.unpack("!i", data[12:16])[0]
        print(x)
        print(y)
        print(height)
        print(ground)
        return x, y, height, ground

    def __get_image(self, data, sling_x, sling_y, sling_height, ground):
        fake_file = io.BytesIO()
        fake_file.write(data)
        im = Image.open(fake_file)

        im.show()
        width, height = im.size

        scale = globalConfig.STD_SLING_HEIGHT / sling_height
        width = int(width*scale)
        height = int(height*scale)
        im = im.resize((width, height))

        sling_x *= scale
        sling_y *= scale
        print (im.size)
        new_im = im.crop((sling_x+300, sling_y-150, sling_x+600, sling_y+150))
        new_im.show()
        new_im.save('sample.jpg')
        print(new_im.size)
        #im = im.convert('L')
        return im

    # Preprocess needs
