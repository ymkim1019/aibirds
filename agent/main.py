import sys
import socket
from PyQt5.QtWidgets import QApplication
from Agent import Agent
from EnvProxy import EnvProxy
from Configuration import globalConfig

def main():
    # Multithreaded Python server : TCP Server Socket Program Stub
    TCP_IP = '0.0.0.0'
    TCP_PORT = 2004

    tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpServer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcpServer.bind((TCP_IP, TCP_PORT))

    if globalConfig.AGENT_TYPE == 'DDPG':
        from DDPGAgent import DDPGAgent
        agent_thread = DDPGAgent()
    elif globalConfig.AGENT_TYPE == 'DQN':
        from DQNAgent import DQNAgent
        agent_thread = DQNAgent()
    else:
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
        new_proxy = EnvProxy(ip, port, conn, agent_thread)
        new_proxy.moveToThread(new_proxy)
        new_proxy.connect_signal()
        new_proxy.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main()
    sys.exit(app.exec_())