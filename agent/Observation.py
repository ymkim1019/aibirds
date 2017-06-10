import struct
import io
from PIL import Image
import numpy as np
from Configuration import globalConfig
import numpy as np

# terminal
class Observation:
    def __init__(self, data):
        split = [4*globalConfig.MAX_BIRDS_SEQ, 4*globalConfig.MAX_BIRDS_SEQ+28]
        self.birds_seq = self.__get_birds_seq(data[:split[0]])
        sling_x, sling_y, sling_height, ground = self.__get_sling(data[split[0]:split[1]])
        #self.img = [0,0,0]
        self.img = self.__get_image(data[split[1]:], sling_x, sling_y, sling_height, ground)

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
        self.pigs_num = struct.unpack("!i", data[16:20])[0]
        self.prev_stars = struct.unpack("!i", data[20:24])[0]
        first_shot = struct.unpack("!i", data[24:28])[0]
        if first_shot == 1 or self.prev_stars != 0:
            self.first_shot = True
        else:
            self.first_shot = False
        return x, y, height, ground

    def __get_image(self, data, sling_x, sling_y, sling_height, ground):
        fake_file = io.BytesIO()
        fake_file.write(data)
        im = Image.open(fake_file)
        #im.show()
        width, height = im.size

        scale = globalConfig.STD_SLING_HEIGHT / sling_height
        width = int(width*scale)
        height = int(height*scale)
        im = im.resize((width, height))

        sling_x *= scale
        sling_y *= scale

        #im = im.convert('L')
        im = im.crop((sling_x+300, sling_y-150, sling_x+600, sling_y+150))
        im = im.resize((globalConfig.OBSERVE_SIZE, globalConfig.OBSERVE_SIZE))
        #im.show()
        #im.save('sample.jpg')

        rgbpix = np.array(im)
        hsvpix = np.array(im.convert('HSV'))
        img = np.zeros((globalConfig.OBSERVE_SIZE, globalConfig.OBSERVE_SIZE), dtype=np.int).tolist()
        for w in range(globalConfig.OBSERVE_SIZE):
            for h in range(globalConfig.OBSERVE_SIZE):
                r, g, b = rgbpix[h][w]
                _, s, v = hsvpix[h][w]

                if (s > 50 and v < 50 and g > 20 and r > 30):  # hill
                    img[h][w] = globalConfig.TYPE_DICT['Hill']
                elif (v > 100 and s < 50):  # stone
                    img[h][w] = globalConfig.TYPE_DICT['Stone']
                elif (r > 150 and g > 100):  # wood
                    img[h][w] = globalConfig.TYPE_DICT['Wood']
                elif (b > 200):  # ice
                    img[h][w] = globalConfig.TYPE_DICT['Ice']
                elif (g > 200):  # pig
                    img[h][w] = globalConfig.TYPE_DICT['Pig']
        return img