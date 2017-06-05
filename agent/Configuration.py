class Configuration:
    def __init__(self):
        self.BUFFER_SIZE = 100000
        self.BATCH_SIZE = 32
        self.GAMMA = 0.99
        self.TAU = 0.001  # Target Network HyperParameters
        self.LRA = 0.0001  # Learning rate for Actor
        self.LRC = 0.001  # Learning rate for Critic

        self.action_dim = 3  # Angle/R/Tap time
        self.state_dim = 10  # 2D input state
        self.EXPLORE = 100000.
        self.episode_count = 2000
        self.max_steps = 100000
        self.epsilon = 1

        self.FRAME_WIDTH = 160
        self.FRAME_HEIGHT = 96
        self.CHANNELS = 1

        self.RedBird = 1 # original : 4
        self.YellowBird = 2 # 5
        self.BlueBird = 3 # 6
        self.BlackBird = 4 # 7
        self.WhiteBird = 5 # 8

globalConfig = Configuration()