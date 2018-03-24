class SlangLexicon:
    def __init__(self):
        words = []
        meanings = []

        filename = open("data/slang-acronyms.csv", "r")
        file = filename.read()
        filename.close()
        file = [line.split(',') for line in file.split("\n")]
        #file = [[int(y) for y in x] for x in poly]

        for i in range(len(file)):
            if not words.__contains__(file[i][0]):
                words.append(file[i][0])
                meanings.append(file[i][1])

        self.meaning_map = dict(zip(words, meanings))

    def getMeaning(self, word):
        try:
            return self.meaning_map[word.upper()]
        except KeyError:
            return 0

    def getSlang(self):
        return self.meaning_map.keys()