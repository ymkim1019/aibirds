import queue
import threading
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)
import numpy as np
from keras.models import Model, load_model
from PIL import Image
import traceback

class brain(threading.Thread):########on-policy trainer#########
	def __init__(self, state_queue,action_queue):
		threading.Thread.__init__(self)
		self.state_queue = state_queue
		self.action_queue = action_queue

	def run(self):
		model = load_model('model2.h5')
		model.summary()
		while True:
			[img,sx,sy,sw,sh] = self.state_queue.get()
			print(sx,sy,sw+sh)
			img = img.crop((240, 180, 840, 480))
			img = img.resize((300, 150), Image.ANTIALIAS)
			im = np.asarray(img)
			im = np.reshape(im, (1,150, 300, 3))
			vec = np.asarray([sx,sy,sw,sh])
			vec = np.reshape(vec,(1,4))
			action_angle = np.argmax(model.predict([im,vec])[0]) - 19 
			self.action_queue.put(action_angle)
