import queue
import threading

class trainer(threading.Thread):
	def __init__(self, queue):
		threading.Thread.__init__(self)
		self.queue = queue
	
	def run(self):
		print(self.queue.get())