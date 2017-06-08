import struct
import copy
from PyQt4.QtCore import QThread, pyqtSignal
from Agent import Agent
import io
from PIL import Image
import numpy as np
from Configuration import globalConfig

class EnvProxy(QThread):
    data_send_requested = pyqtSignal()

    def __init__(self, ip, port, conn, agent):
        super(EnvProxy, self).__init__()

        self.ip = ip
        self.port = port
        self.sock_conn = conn
        self.agent = agent
        self.verbose = True
        self.send_buf = list()
        self.last_n_pigs = None
        self.last_n_ices = None

        if self.verbose:
            print("[+] New server socket thread started for " + ip + ":" + str(port))

    def get_client_ip(self):
        return self.ip

    def connect_signal(self):
        self.data_send_requested.connect(self.on_data_send_requested)

    def on_data_send_requested(self):
        print("on_data_send_requested..")
        while len(self.send_buf) > 0:
            buf = self.send_buf.pop(0)
            try:
                self.socket_conn.send(buf)
            except Exception as e:
                print(e)
                self.exit(0)

    def execute(self, target, high_shot, tap_time):
        # print('execute ', target, high_shot, tap_time)
        data = struct.pack("!i", target) + struct.pack("!i", high_shot) + struct.pack("!i", tap_time)
        self.send_data(Agent.ACT, data)

    def execute(self, target_x, target_y, angle, tap_time):
        data = struct.pack("!i", target_x) + struct.pack("!i", target_y) + struct.pack("!d", angle) + struct.pack("!i", tap_time)
        self.send_data(Agent.ACT, data)

    def send_data(self, job_id, data):
        # print("send_data..job_id =", job_id, " data len =", len(data))
        buf = struct.pack("!i", job_id) + data
        size = struct.pack("!i", len(buf))
        self.sock_conn.send(size+buf)

    def recv_all(self, size):
        msg = b''
        while len(msg) < size:
            part = self.sock_conn.recv(size - len(msg))
            if part == b'':
                break  # the connection is closed
            msg += part
        return msg

    def run(self):
        while True:
            try:
                length = self.recv_all(4)
            except Exception as e:
                print(str.format("Comthread : {}:{}", self.ip, self.port))
                print(e)
                self.exit(0)
            size = struct.unpack("!i", length)[0]
            print("size:", size)
            tot_data = self.recv_all(size)

            job_id = struct.unpack("!i", tot_data[0:4])[0]
            copied_data = copy.deepcopy(tot_data[4:])

            if self.verbose:
                print(str.format("Server received data from {}:{}, job_id={}", self.ip, self.port, job_id))

            if job_id == Agent.OBSERVE:
                if globalConfig.agent_type == 'dqn':
                    actions = list();

                    is_first_shot = True if struct.unpack("!i", copied_data[0:4])[0] == 1 else False
                    done = struct.unpack("!i", copied_data[4:8])[0]
                    n_stars = struct.unpack("!i", copied_data[8:12])[0]
                    n_pigs = struct.unpack("!i", copied_data[12:16])[0]
                    n_stones = struct.unpack("!i", copied_data[16:20])[0]
                    n_woods = struct.unpack("!i", copied_data[20:24])[0]
                    n_ices = struct.unpack("!i", copied_data[24:28])[0]
                    n_tnts = struct.unpack("!i", copied_data[28:32])[0]
                    bird_type = struct.unpack("!i", copied_data[32:36])[0]
                    current_level = struct.unpack("!i", copied_data[36:40])[0]
                    sling_x = struct.unpack("!i", copied_data[40:44])[0]
                    sling_y = struct.unpack("!i", copied_data[44:48])[0]
                    n_actions = struct.unpack("!i", copied_data[48:52])[0]
                    for i in range(n_actions):
                        start_pos = 52 + i * 20
                        target_type = struct.unpack("!i", copied_data[start_pos:start_pos + 4])[0]
                        x = struct.unpack("!i", copied_data[start_pos + 4:start_pos + 8])[0]
                        y = struct.unpack("!i", copied_data[start_pos + 8:start_pos + 12])[0]
                        angle = struct.unpack("!d", copied_data[start_pos + 12:start_pos + 20])[0]
                        actions.append([target_type, x, y, angle])

                    fake_file = io.BytesIO()
                    fake_file.write(copied_data[52+n_actions*20:])
                    im = Image.open(fake_file)
                    im = im.crop((400, 40, 800, 440))
                    im = im.resize((globalConfig.FRAME_WIDTH, globalConfig.FRAME_HEIGHT), Image.NEAREST)
                    # im.show()
                    im = np.array(im.convert('L'))
                    im = im[:, :, np.newaxis]

                    r = -1 # one shot
                    if is_first_shot is True:
                        if n_stars == 0:
                            r += -10 # level failed
                        else:
                            r += n_stars * 5
                    else:
                        r += (self.last_n_pigs - n_pigs) * 2 + (self.last_n_ices - n_ices) * 0.1

                    self.last_n_pigs = n_pigs
                    self.last_n_ices = n_ices
                    ob = (is_first_shot, done, n_pigs, n_stones, n_woods, n_ices, n_tnts, bird_type
                          , sling_x, sling_y, actions, im, r, current_level)
                    self.agent.add_job(job_id, ob, self)
                else:
                    is_first_shot = True if struct.unpack("!i", copied_data[0:4])[0] == 1 else False
                    done = struct.unpack("!i", copied_data[4:8])[0]
                    n_stars = struct.unpack("!i", copied_data[8:12])[0]
                    n_pigs = struct.unpack("!i", copied_data[12:16])[0]
                    n_stones = struct.unpack("!i", copied_data[16:20])[0]
                    n_woods = struct.unpack("!i", copied_data[20:24])[0]
                    n_ices = struct.unpack("!i", copied_data[24:28])[0]
                    n_tnts = struct.unpack("!i", copied_data[28:32])[0]
                    bird_type = struct.unpack("!i", copied_data[32:36])[0]
                    current_level = struct.unpack("!i", copied_data[36:40])[0]
                    fake_file = io.BytesIO()
                    fake_file.write(copied_data[40:])
                    im = Image.open(fake_file)
                    im = im.resize((globalConfig.FRAME_WIDTH, globalConfig.FRAME_HEIGHT), Image.NEAREST)
                    # im.show()
                    im = np.array(im.convert('L'))
                    im = im[:, :, np.newaxis]

                    r = -1 # one shot
                    if is_first_shot is True:
                        if n_stars == 0:
                            r += -10 # level failed
                        else:
                            r += n_stars * 3
                    else:
                        r += (self.last_n_pigs - n_pigs) * 2

                    self.last_n_pigs = n_pigs
                    ob = (is_first_shot, done, n_pigs, n_stones, n_woods, n_ices, n_tnts, bird_type, im, r, current_level)
                    self.agent.add_job(job_id, ob, self)
            else:
                self.agent.add_job(job_id, copied_data, self)