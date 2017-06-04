import struct
import copy
from PyQt5.QtCore import QThread, pyqtSignal


class ComThread(QThread):
    data_send_requested = pyqtSignal()

    def __init__(self, ip, port, conn, agent):
        super(ComThread, self).__init__()

        self.ip = ip
        self.port = port
        self.sock_conn = conn
        self.agent = agent
        self.verbose = True
        self.send_buf = list()

        if self.verbose:
            print("[+] New server socket thread started for " + ip + ":" + str(port))

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

    def send_data(self, job_id, data):
        print("send_data..")
        buf = struct.pack("i", job_id) + data
        size = struct.pack("i", len(buf))

        self.sock_conn.send(size+buf)

        # self.send_buf.append(size+buf)
        # self.data_send_requested.emit()

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
                print(str.format("Server received data from {}:{}", self.ip, self.port))

            self.agent.add_job(job_id, copied_data, self)



