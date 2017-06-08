from PyQt4.QtCore import QThread, pyqtSignal, QCoreApplication


class EventTask(QThread):
    job_arrived = pyqtSignal(tuple)

    def __init__(self, name='EventTask'):
        super(EventTask, self).__init__()

        self.name = name
        self.verbose = 0
        self.job_queue = list()

    def connect_signal(self):
        self.job_arrived.connect(self.do_job)

    def do_job(self, job_obj):
        (job_id, data, com_thread) = job_obj
        if self.verbose:
            print(str.format("Processing Job id:{}..", job_id))

    def add_job(self, job_id, data, com_thread=None):
        if self.verbose:
            print(str.format("Job id:{} arrived", job_id))
        self.job_queue.append((job_id, data, com_thread))

    def run(self):
        if self.verbose:
            print(str.format("task:{} started..", self.name))

        cnt = 0

        while True:
            if len(self.job_queue) > 0:
                # print("pop")
                obj = self.job_queue.pop(0)
                self.job_arrived.emit(obj)

            self.msleep(10)
            cnt += 1
            if cnt % 300 == 0:
                cnt = 0
                self.add_job(1, 0)


