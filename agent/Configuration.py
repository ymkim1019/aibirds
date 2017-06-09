class Configuration:
    def __init__(self):
        self.TYPE = {0: 'Unknown', 1: 'Ground', 2: 'Hill', 3: 'Sling', 4: 'RedBird', 5: 'YellowBird', 6: 'BlueBird',
                     7: 'BlackBird', 8: 'WhiteBird', 9: 'Pig', 10: 'Ice', 11: 'Wood', 12: 'Stone', 18: 'TNT'}
        self.TYPE_DICT = {}
        for k in self.TYPE.keys():
            self.TYPE_DICT[self.TYPE[k]] = k
        self.MAX_BIRDS_SEQ = 10


        self.SHOW = False
        self.AGENT_TYPE = 'RETRACE'
        #self.AGENT_TYPE = 'DQN'
        self.STD_SLING_HEIGHT = 60
        self.OBSERVE_SIZE = 84

globalConfig = Configuration()