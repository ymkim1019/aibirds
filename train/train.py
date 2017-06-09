import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)
from keras.models import Model, Sequential
from keras.layers import Convolution2D, Flatten, Dense, pooling, Input, concatenate
import json
from PIL import Image
import math
import numpy as np
import scipy
import random
# model = Sequential()
# model.add(pooling.MaxPooling2D(pool_size=(3,3), input_shape=(150,300,3)))
# model.add(Convolution2D(8,(9,9),strides=(3,3),activation='relu'))
# model.add(pooling.MaxPooling2D(pool_size=(2,2)))
# model.add(Convolution2D(8,(5,5),strides=(2,2),activation='relu'))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(100))
image_input = Input(shape=(150,300,3))
vector_input = Input(shape=(4,))

conv = pooling.MaxPooling2D(pool_size=(3,3))(image_input)
conv = Convolution2D(8,(9,9),strides=(3,3),activation='relu')(conv)
conv = pooling.MaxPooling2D(pool_size=(2,2))(conv)
conv = Convolution2D(8,(5,5),strides=(2,2),activation='relu')(conv)
feature = Flatten()(conv)
feature = concatenate([feature,vector_input])
feature = Dense(64)(feature)
feature = Dense(64)(feature)
Q_value = Dense(100)(feature)
model = Model([image_input,vector_input],Q_value)

#im = im.crop((240,180,840,480))
#state.show()
#im = im.resize((300,150),Image.ANTIALIAS)
model.compile(loss='mse',optimizer='adam')
model.summary();

batch_size = 10
gamma = 0.8
min_angle = [80] * 63
max_angle = [-19] * 63
angle_range = [0] * 63
level_shot_num = [0] * 63
json_file = open("history.json").read()
memory = json.loads(json_file)

def angle(step):
	dx = step[13]
	dy = step[14]
	real_angle = math.degrees(math.atan(-dy/dx))
	action_angle = round(min(80,max(-19,real_angle)))
	return action_angle

i= 0
while i <len(memory):
	episode = memory[i]
	if (len(episode[-2]) == 1):
		memory.pop(i)
		continue
	i += 1

for episode in memory:
	for step in episode[:-1]:
		level = step[0]
		min_angle[level-1] = min(min_angle[level-1], angle(step))
		max_angle[level-1] = max(max_angle[level-1], angle(step))
		angle_range[level-1] = max_angle[level-1] - min_angle[level-1] + 1
		level_shot_num[level-1] += 1


for _ in range(len(memory) * 10):
	episode_list = random.sample(memory,batch_size)
	for episode in episode_list:
		for i in range(len(episode)-2):
			step = episode[i]
			action = angle(step) + 19
			next_step = episode[i+1]
			try:
				img = Image.open("{}_{}_{}.jpg".format(step[0],step[1],step[2]))
				img = img.crop((240, 180, 840, 480))
				img = img.resize((300,150),Image.ANTIALIAS)
				im = np.asarray(img)
				im = np.reshape(im, (1,150, 300, 3))
				next_img = Image.open("{}_{}_{}.jpg".format(next_step[0],next_step[1],next_step[2]))
				next_img = next_img.crop((240,180,840,480))
				next_img = next_img.resize((300,150),Image.ANTIALIAS)
				next_im = np.asarray(next_img)
				next_im = np.reshape(next_img,(1,150,300,3))
			except:
				break
			vec = np.asarray([step[3],step[4],step[5],step[6]])
			next_vec = np.asarray([next_step[3], next_step[4], next_step[5], next_step[6]])
			vec = np.reshape(vec,(1,4))
			next_vec = np.reshape(next_vec, (1, 4))
			target = -1 + step[6] - episode[i+1][6] + gamma * np.amax(model.predict([next_im,next_vec])[0])
			target_f = model.predict([im,vec])
			target_f[0][action] = target
			model.fit([im,vec],target_f,epochs=1,verbose=0)
			for _ in range(np.random.poisson((100-angle_range[step[0]])/angle_range[step[0]])):
				angle = 0
				while(True):
					angle = random.randint(-19,80)
					if(angle<min_angle[step[0]] or angle>max_angle[step[0]]):
						break
				action = angle(step) + 19
				target = -1 + gamma * np.amax(model.predict([im, vec])[0])
				target_f = model.predict([im, vec])
				target_f[0][action] = target
				model.fit([im, vec], target_f, epochs=1, verbose=0)
			img.close()
			next_img.close()

		#last step

		step = episode[-2]
		action = angle(step) + 19
		score = episode[-1][0]
		try:
			img = Image.open("C:\\Users\\임수현\\Desktop\\angrybird_image\\{}_{}_{}.jpg".format(step[0], step[1], step[2]))
			img = img.crop((240, 180, 840, 480))
			img = img.resize((300, 150), Image.ANTIALIAS)
		except:
			break
		im = np.asarray(img)
		im = np.reshape(im, (1,150, 300, 3))
		vec = np.asarray([step[3], step[4], step[5], step[6]])
		vec = np.reshape(vec, (1, 4))
		if (score>0):
			target = -1 + step[6]
		else:
			target = -1
		target_f = model.predict([im, vec])
		target_f[0][action] = target
		model.fit([im, vec], target_f, epochs=1, verbose=0)
		for _ in range(np.random.poisson((100 - angle_range[step[0]-1]) / angle_range[step[0]-1])):
			real_angle = 0
			while (True):
				real_angle = random.randint(-19, 80)
				if (real_angle < min_angle[step[0]-1] or real_angle > max_angle[step[0]-1]):
					break
			action = angle(step) + 19
			target = -1
			target_f = model.predict([im, vec])
			target_f[0][action] = target
			model.fit([im, vec], target_f, epochs=1, verbose=0)
		img.close()

	model.save("model.h5")
#






