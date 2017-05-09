import sys
import socket
from PyQt4 import QtGui
from Agent import Agent
from ComThread import ComThread

def main():
    # Multithreaded Python server : TCP Server Socket Program Stub
    TCP_IP = '0.0.0.0'
    TCP_PORT = 2004

    tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpServer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcpServer.bind((TCP_IP, TCP_PORT))

    agent_thread = Agent()
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
        new_thread = ComThread(ip, port, conn, agent_thread)
        new_thread.moveToThread(new_thread)
        new_thread.connect_signal()
        new_thread.start()

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    main()
    sys.exit(app.exec_())