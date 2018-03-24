from collections import OrderedDict

class PopCounter:
    def __init__(self, tweets):
        self.frequency_map = {}

        for t in tweets:
            for token in t:
                if token in self.frequency_map.keys():
                    self.frequency_map[token] += 1
                else:
                    self.frequency_map[token] = 1


    def print(self):

        od = OrderedDict(sorted(self.frequency_map.items(), key=lambda t: -t[1]))
        for k, v in od.items():
            print(k+','+str(v))

