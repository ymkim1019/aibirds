import numpy as np
from EventTask import EventTask
import tensorflow as tf
from ReplayBuffer import ReplayBuffer
from QvalueNetwork import QvalueNetwork
import math
import keras.backend as K
import json
from Configuration import globalConfig
from PyQt4.QtCore import QTimer
import pickle


class AgentDDQN(EventTask):
    # ENV -> AGENT
    OBSERVE = 0
    REPLAY = 1

    # AGENT -> ENV
    ACT = 0

    def __init__(self, trainable=1, load_model=1):
        super(AgentDDQN, self).__init__('Agent')

        np.random.seed(9874)

        self.step = 0
        self.state_cache = dict()
        self.action_cache = dict()
        # self.episode_dict = dict()

        # Tensorflow GPU optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.trainable = trainable
        K.set_session(self.sess)

        self.network = QvalueNetwork(self.sess, globalConfig.TAU, globalConfig.LRC)

        self.buff = ReplayBuffer(globalConfig.BUFFER_SIZE)  # Create replay buffer
        self.cnt = 0

        if load_model == 1:
            # Now load the weight
            print("Now we load the weight")
            try:
                self.network.model.load_weights("ddqnmodel.h5")
                self.network.target_model.load_weights("ddqnmodel.h5")
                print("Weight load successfully")
            except:
                print("Cannot find the weight")

        self.graph = tf.get_default_graph()
        self.timer = None

        globalConfig.epsilon = max(globalConfig.epsilon - (self.buff.count() // globalConfig.epsilon_decay_interval) * globalConfig.epsilon_decay, globalConfig.epsilon_min)

    def start_timer(self):
        if self.trainable == 1:
            print('start qtimer')
            self.timer = QTimer()
            self.timer.timeout.connect(self.timer_fired)
            self.timer.start(1000)

    def timer_fired(self):
        print('timer fired...')
        self.add_job(self.REPLAY, 0)

    #
    # def print_episode(self, episode):
    #     print('---------------episode----------------')
    #     for each in episode:
    #         s_t = each[0]
    #         a_t = each[1]
    #         r_t = each[2]
    #         s_t1 = each[3]
    #         done = each[4]
    #         print(str.format("s_t(num_objects:{}, bird:{}, sling:({},{}), a_t({}, angle={}, tap={})"
    #                          ", s_t_1(num_objects:{}, bird:{}, sling:({},{}), r_t({}), done({})",
    #               s_t[1], s_t[2], s_t[3][0], s_t[3][1],
    #               globalConfig.target_type_strings[int(a_t[0])], a_t[1], int(a_t[2]),
    #               s_t1[1], s_t1[2], s_t1[3][0], s_t1[3][1],
    #               r_t, done))
    #     print('--------------------------------------')

    def replay(self):
        if self.buff.count() > 0 and self.trainable == 1:
            self.cnt += 1

            print("Do the batch update...# of experiences =", self.buff.count())

            # (self.state_cache[env_proxy], self.action_cache[env_proxy], r_t, s_t1, new_actions, done)

            # Do the batch update
            batch = self.buff.getBatch(globalConfig.BATCH_SIZE)

            # s_t
            images = [e[0][0] for e in batch]
            num_objects = [e[0][1] for e in batch]
            birds = [e[0][2] for e in batch]
            slings = [e[0][3] for e in batch]
            states = [np.array(images), np.array(num_objects), np.array(birds), np.array(slings)]

            # a_t
            a_t = np.asarray([e[1] for e in batch])

            # r_t
            rewards = np.asarray([e[2] for e in batch])

            # s_t1
            new_images = [e[3][0] for e in batch]
            new_num_objects = [e[3][1] for e in batch]
            new_birds = [e[3][2] for e in batch]
            new_slings = [e[3][3] for e in batch]

            # a_t1
            a_t1_candidates = [e[4] for e in batch]

            # dones
            dones = np.asarray([e[5] for e in batch])

            # just placeholder
            y_t = np.zeros(len(batch))
            target_q_values = np.zeros(len(batch))
            a_t1s = []

            for i in range(len(batch)):
                new_states = [np.array([new_images[i]] * len(a_t1_candidates[i]))
                    , np.array([new_num_objects[i]] * len(a_t1_candidates[i]))
                    , np.array([new_birds[i]] * len(a_t1_candidates[i]))
                    , np.array([new_slings[i]] * len(a_t1_candidates[i]))]
                q_values = self.network.model.predict(new_states + [np.array(a_t1_candidates[i])])
                a_t1s.append(np.array(a_t1_candidates[i][np.argmax(np.ravel(q_values))]))

            new_states = [np.array(new_images), np.array(new_num_objects), np.array(new_birds), np.array(new_slings)]
            target_q_values = self.network.target_model.predict(new_states+[np.array(a_t1s)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + globalConfig.GAMMA * target_q_values[k][0]

            print('loss =', self.network.model.train_on_batch(states + [a_t], y_t))

            if self.cnt % globalConfig.target_update_interval == 0:
                print("Target train....")
                self.network.target_train()
                self.cnt = 0
                print("Saving weights....")
                self.network.target_model.save_weights("ddqnmodel.h5", overwrite=True)

    # def run(self):
    #     self.start_timer()
    #     super(AgentDQN, self).run()

    def do_job(self, job_obj):
        (job_id, data, env_proxy) = job_obj
        if self.verbose:
            print(str.format("Processing Job id:{}..", job_id))

        with self.graph.as_default():
            if job_id == self.REPLAY:
                self.replay()
            elif job_id == self.OBSERVE:
                # observation
                is_first_shot, done, n_pigs, n_stones, n_woods, n_ices, n_tnts, bird_type \
                    , sling_x, sling_y, actions, im, r_t, current_level = data
                print(str.format("----> observation from {}, level = {}", env_proxy.get_client_ip(), current_level))
                print(str.format("first shot:{}, reward:{}, episode done:{}", is_first_shot, r_t, done))
                print(str.format(
                    "# pigs={}, # stones={}, # woods={}, # ices={}, n_tnts={}, bird={}, sling=({}, {}), n_actions={}"
                    , n_pigs, n_stones, n_woods, n_ices, n_tnts, bird_type, sling_x, sling_y, len(actions)))
                # print('im shape=', im.shape)
                s_t1 = [np.array(im), np.array([n_pigs, n_stones, n_woods, n_ices, n_tnts]), np.array([bird_type]),
                        np.array([sling_x, sling_y])]

                # replay
                # self.replay()

                # select action a_t
                s_t = s_t1
                pixels = np.array([s_t[0]] * len(actions))
                num_objects = np.array([s_t[1]] * len(actions))
                input_bird = np.array([s_t[2]] * len(actions))
                sling = np.array([s_t[3]] * len(actions))
                states = [pixels, num_objects, input_bird, sling]
                new_actions = []
                target_coordinates = []
                for a in actions:
                    for tap in globalConfig.tap_values[bird_type]:
                        # print(str.format("target_type = {}, ({}, {}), angle={}, tap={}"
                        #                  , globalConfig.target_type_strings[int(a[0])], int(a[1]), int(a[2]), a[3], tap))
                        new_actions.append(np.array([a[0], a[3], tap]))
                        target_coordinates.append([a[1], a[2]])
                new_actions = np.array(new_actions)
                q_values = self.network.target_model.predict(states + [new_actions])
                sorted_indexes = [k[0] for k in sorted(enumerate(q_values), reverse=True, key=lambda x: x[1])]
                # print(target_q_values[sorted_indexes])

                print('cnt=', self.cnt)
                if self.cnt % globalConfig.epsilon_decay_interval == 0:
                    old = globalConfig.epsilon
                    globalConfig.epsilon = max(globalConfig.epsilon - globalConfig.epsilon_decay, globalConfig.epsilon_min)
                    print(str.format('epsilon decay from {} to {}', old, globalConfig.epsilon))

                rand_num = np.random.rand()
                print('rand_num =', rand_num, ' epsilon =', globalConfig.epsilon)

                if self.trainable == 1 and rand_num < globalConfig.epsilon:
                    action_idx = 1 + np.random.randint(len(sorted_indexes)-1)
                    print("randomly select an action except the best one..best_q_value ="
                          , q_values[sorted_indexes[0]]
                          , " chosen q_value =", q_values[sorted_indexes[action_idx]])
                else:
                    print("choose the best action..q_value =", q_values[sorted_indexes[0]])
                    action_idx = 0
                a_t = new_actions[sorted_indexes[action_idx]]

                target_x = target_coordinates[action_idx][0]
                target_y = target_coordinates[action_idx][1]
                target_type = int(a_t[0])
                angle = a_t[1]
                tap_time = int(a_t[2])
                print(str.format("next action : target({}:{},{}), angle({}), tap time({})"
                                 , globalConfig.target_type_strings[target_type], target_x, target_y, angle, tap_time))

                # add trajectory
                try:
                    self.buff.add_dqn_exp(self.state_cache[env_proxy], self.action_cache[env_proxy], r_t, s_t1
                                  , new_actions, done)
                    print("store transition into replay buffer")

                    # print("add trajectory")
                    # self.episode_dict[env_proxy.get_client_ip()].append([self.state_cache[env_proxy], self.action_cache[env_proxy], r_t, s_t1, done])
                    # if done == 1:
                    #     print("store transition into replay buffer")
                    #     self.buff.add(self.episode_dict[env_proxy.get_client_ip()])
                    #     self.print_episode(self.episode_dict[env_proxy.get_client_ip()])
                    #     self.episode_dict.pop(env_proxy.get_client_ip(), None)
                except Exception as e:
                    print(e)
                    print("first shot of the game")
                    # self.episode_dict[env_proxy.get_client_ip()] = list()
                    pass

                # cache
                self.state_cache[env_proxy] = s_t
                self.action_cache[env_proxy] = a_t

                # execute an action
                env_proxy.execute(target_x, target_y, angle, tap_time)