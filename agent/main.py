import sys
import socket
from PyQt4 import QtGui
from Agent import Agent
from EnvProxy import EnvProxy
import argparse

def main():
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-trainable", "--trainable", default=1, type=int)
    parser.add_argument("-load_model", "--load_model", default=1, type=int)

    args = parser.parse_args()
    print(args)

    # Multithreaded Python server : TCP Server Socket Program Stub
    TCP_IP = '0.0.0.0'
    TCP_PORT = 2004

    tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpServer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcpServer.bind((TCP_IP, TCP_PORT))

    agent_thread = Agent(trainable=args.trainable, load_model=args.load_model)
    agent_thread.moveToThread(agent_thread)
    agent_thread.connect_signal()
    agent_thread.start()

    while True:
        # for the test purpose
        # import time
        # time.sleep(3)
        # agent_thread.add_job(1, "abc")

        tcpServer.listen(4)
        print("Multi-threaded Python server : Waiting for connections from TCP clients...")
        (conn, (ip, port)) = tcpServer.accept()
        new_proxy = EnvProxy(ip, port, conn, agent_thread)
        new_proxy.moveToThread(new_proxy)
        new_proxy.connect_signal()
        new_proxy.start()

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    main()
    sys.exit(app.exec_())