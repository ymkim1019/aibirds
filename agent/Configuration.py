import numpy as np

class Configuration:
    def __init__(self):
        self.BUFFER_SIZE = 100000
        self.BATCH_SIZE = 32
        self.GAMMA = 0.99
        self.TAU = 0.001  # Target Network HyperParameters
        self.LRA = 0.0001  # Learning rate for Actor
        self.LRC = 0.001  # Learning rate for Critic

        self.action_dim = 3  # target/high_low/Tap time
        self.EXPLORE = 100000.
        self.episode_count = 2000
        self.max_steps = 100000
        self.epsilon = 1

        self.FRAME_WIDTH = 80
        self.FRAME_HEIGHT = 80
        self.CHANNELS = 1

        self.RedBird = 1 # original : 4
        self.YellowBird = 2 # 5
        self.BlueBird = 3 # 6
        self.BlackBird = 4 # 7
        self.WhiteBird = 5 # 8
        self.model_save_interval = 5

        self.TARGET_PIG = 0
        self.TARGET_STONE = 1
        self.TARGET_WOOD = 2
        self.TARGET_ICE = 3
        self.TARGET_TNT = 4

        self.agent_type = ""

        self.tap_values = [
            [0], # red
            list(np.arange(65, 91, 5)), # yellow
            list(np.arange(70, 91, 5)), # blue
            list(np.arange(70, 91, 5)), # black
            list(np.arange(65, 86, 5)) # white
        ]
        self.target_type_strings = ["pig", "stone", "wood", "ice", "tnt"]
        self.ep_greedy = 0.3


globalConfig = Configuration()