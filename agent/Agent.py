import numpy as np
from EventTask import EventTask
import tensorflow as tf
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
from Configuration import globalConfig
import math
import keras.backend as K
import json


class Agent(EventTask):
    # ENV -> AGENT
    OBSERVE = 0

    # AGENT -> ENV
    ACT = 0

    def __init__(self, trainable=1):
        super(Agent, self).__init__('Agent')

        np.random.seed(1337)

        self.step = 0
        self.state_cache = dict()
        self.action_cache = dict()

        # Tensorflow GPU optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.trainable = trainable
        K.set_session(self.sess)

        self.actor = ActorNetwork(self.sess, globalConfig.TAU, globalConfig.LRA)
        self.critic = CriticNetwork(self.sess, globalConfig.TAU, globalConfig.LRC)

        self.buff = ReplayBuffer(globalConfig.BUFFER_SIZE)  # Create replay buffer
        self.cnt = 0

        # Now load the weight
        print("Now we load the weight")
        try:
            self.actor.model.load_weights("actormodel.h5")
            self.critic.model.load_weights("criticmodel.h5")
            self.actor.target_model.load_weights("actormodel.h5")
            self.critic.target_model.load_weights("criticmodel.h5")
            print("Weight load successfully")
        except:
            print("Cannot find the weight")

        self.graph = tf.get_default_graph()


    def do_job(self, job_obj):
        (job_id, data, env_proxy) = job_obj
        if self.verbose:
            print(str.format("Processing Job id:{}..", job_id))

        with self.graph.as_default():
            if job_id == self.OBSERVE:
                # observation
                is_first_shot, done, n_pigs, n_stones, n_woods, n_ices, n_tnts, bird_type, im, r_t = data
                print(str.format("----> observation from {}", env_proxy.get_client_ip()))
                print(str.format("first shot:{}, reward:{}, episode done:{}", is_first_shot, r_t, done))
                print(str.format("# pigs={}, # stones={}, # woods={}, # ices={}, n_tnts={}, bird={}"
                                 , n_pigs, n_stones, n_woods, n_ices, n_tnts, bird_type))
                # print('im shape=', im.shape)
                s_t1 = [np.array(im), np.array([n_pigs, n_stones, n_woods, n_ices, n_tnts]), np.array([bird_type])]

                # store transition into replay buffer
                try:
                    self.buff.add(self.state_cache[env_proxy], self.action_cache[env_proxy], r_t, s_t1, done)
                    print("store transition into replay buffer")
                except Exception as e:
                    print("first shot of the game")
                    pass

                if self.buff.count() > 0:
                    print("Do the batch update...")

                    # Do the batch update
                    batch = self.buff.getBatch(globalConfig.BATCH_SIZE)
                    # states = np.asarray([e[0] for e in batch])
                    images = [e[0][0] for e in batch]
                    num_objects = [e[0][1] for e in batch]
                    birds = [e[0][2] for e in batch]
                    states = [np.array(images), np.array(num_objects), np.array(birds)]
                    actions = np.asarray([e[1] for e in batch])
                    rewards = np.asarray([e[2] for e in batch])
                    new_images = [e[3][0] for e in batch]
                    new_num_objects = [e[3][1] for e in batch]
                    new_birds = [e[3][2] for e in batch]
                    new_states = [np.array(new_images), np.array(new_num_objects), np.array(new_birds)]
                    dones = np.asarray([e[4] for e in batch])
                    y_t = np.asarray([e[1] for e in batch])

                    # print('batch update shape, size =', len(batch))
                    # print(np.array(images).shape)
                    # print(np.array(num_objects).shape)
                    # print(np.array(birds).shape)

                    new_a = self.actor.target_model.predict(states)
                    # print('new_a=\n', new_a)
                    target_q_values = self.critic.target_model.predict(new_states + [new_a])
                    # print('q values =\n', target_q_values)

                    for k in range(len(batch)):
                        if dones[k]:
                            y_t[k] = rewards[k]
                        else:
                            y_t[k] = rewards[k] + globalConfig.GAMMA * target_q_values[k]

                    if self.trainable:
                        print('loss =', self.critic.model.train_on_batch(states + [actions], y_t))
                        a_for_grad = self.actor.model.predict(states)
                        grads = self.critic.gradients(states, a_for_grad)
                        self.actor.train(states, grads)
                        self.actor.target_train()
                        self.critic.target_train()

                # select action a_t
                s_t = s_t1
                pixels = np.reshape(s_t[0], tuple([1]) + s_t[0].shape)
                num_objects = np.reshape(s_t[1], tuple([1]) + s_t[1].shape)
                input_bird = np.reshape(s_t[2], tuple([1]) + s_t[2].shape)

                # print(pixels.shape, num_objects.shape, input_bird.shape)

                a_t = self.actor.model.predict([pixels, num_objects, input_bird])
                target = math.floor(a_t[0][0] * np.sum(s_t[1]))
                high_shot = 1 if a_t[0][1] > 0.5 else 0
                tap_time = math.floor(65 + a_t[0][2] * 25)
                print('raw a_t =', a_t)
                print(str.format("next action: target({}), high_shot({}), tap time({})", target, high_shot, tap_time))

                # cache
                self.state_cache[env_proxy] = s_t
                self.action_cache[env_proxy] = a_t[0]

                # execute an action
                env_proxy.execute(target, high_shot, tap_time)

                self.cnt += 1
                if self.cnt % globalConfig.model_save_interval == 0:
                    if self.trainable:
                        print("Saving mode....")
                        self.actor.model.save_weights("actormodel.h5", overwrite=True)
                        with open("actormodel.json", "w") as outfile:
                            json.dump(self.actor.model.to_json(), outfile)

                        self.critic.model.save_weights("criticmodel.h5", overwrite=True)
                        with open("criticmodel.json", "w") as outfile:
                            json.dump(self.critic.model.to_json(), outfile)
