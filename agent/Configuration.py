class Configuration:
    def __init__(self):
        self.TYPE = {0: 'Unknown', 1: 'Ground', 2: 'Hill', 3: 'Sling', 4: 'RedBird', 5: 'YellowBird', 6: 'BlueBird',
                     7: 'BlackBird', 8: 'WhiteBird', 9: 'Pig', 10: 'Ice', 11: 'Wood', 12: 'Stone', 18: 'TNT'}
        self.MAX_BIRDS_SEQ = 10
        self.SHOW = False
        self.AGENT_TYPE = 'NORMAL'
        #self.AGENT_TYPE = 'DQN'
        self.STD_SLING_HEIGHT = 60

        self.MAX_ANGLE = 90
        self.MIN_ANGLE = 0
        self.ANGLE_UNIT = 1

        if self.ANGLE_UNIT == 0:
            self.ACTION_NUM = 0
        else:
            self.ACTION_NUM = (self.MAX_ANGLE - self.MIN_ANGLE + 1)/self.ANGLE_UNIT

        self.GAMMA = 0.9
        self.INITIAL_EPSILON = 1.0

globalConfig = Configuration()