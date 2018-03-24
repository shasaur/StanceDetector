from porterstemmer import PorterStemmer
import collections

class MPQALexicon:
    def __init__(self):
        stemmer = PorterStemmer()

        words = []
        polarities = []

        filename = open("data/subjclueslen1-HLTEMNLP05.tff", "r")
        file = filename.read()
        filename.close()
        file = [line.split(' ') for line in file.split("\n")]
        #file = [[int(y) for y in x] for x in poly]

        for i in range(len(file)):
            word = file[i][2].split('=')[1]
            type = file[i][0].split('=')[1]
            priorpolarity = file[i][5].split('=')[1]

            multiplier = (2 if type=="strongsubj" else 1)
            if priorpolarity == "positive":
                value = 1
            elif priorpolarity == "negative":
                value = -1
            else:
                value = 0
            #value = (1 if priorpolarity == "positive" else -1)

            words.append(word)
            polarities.append(multiplier*value)

        new_words = []
        for w_i in range(len(words)):
            word = words[w_i]
            new_words.append(stemmer.stem(word, 0, len(word) - 1))

        ## Build stemmed subjectivity index
        self.polarity_map = dict(zip(words, polarities))
        #self.stem_map = dict(zip(new_words, word))
        new_words_zip = list(zip(new_words, polarities))
        new_words_frequencies = collections.Counter(new_words_zip)
        new_words_frequencies_keys = list(new_words_frequencies.keys())
        new_words_frequencies_values = list(new_words_frequencies.values())


        self.stem_polarity_map = {}
        for w_i in range(len(new_words_frequencies_keys)):
            search_term = new_words_frequencies_keys[w_i][0]
            if not search_term in self.stem_polarity_map.keys():
                commons = []
                short_sight_index = w_i
                while True:
                    commons.append([new_words_frequencies_keys[short_sight_index][1], new_words_frequencies_values[short_sight_index]])
                    if short_sight_index+1 < len(new_words_frequencies_keys):
                        short_sight_index += 1
                        next_word = new_words_frequencies_keys[short_sight_index][0]
                        if not next_word == search_term:
                            break
                    else:
                        break

                ## LOGIC

                ## following could be done with sorting
                if len(commons) > 1:
                    # check whether there's contrasting polarity signs
                    positives = []
                    neutrals = []
                    negatives = []
                    current_class = -1
                    conflict = False

                    for c in commons:
                        if c[0] > 0:
                            positives.append(c)

                            if current_class == -1:
                                current_class = 0
                            elif not current_class == 0:
                                conflict = True

                        elif c[0] == 0:
                            neutrals.append(c)

                            if current_class == -1:
                                current_class = 1
                            elif not current_class == 1:
                                conflict = True
                        else:
                            negatives.append(c)

                            if current_class == -1:
                                current_class = 2
                            elif not current_class == 2:
                                conflict = True

                    if conflict:
                        self.stem_polarity_map[search_term] = 0
                    # if not, we can check which is the dominating polarity
                    else:
                        ## really bad honestly, just determining which array to use
                        mains = positives
                        if len(negatives) > 0:
                            mains = negatives
                        elif len(neutrals) > 0:
                            mains = neutrals

                        # Determine dominating polarity
                        highest_polarity = -10
                        highest_polarity_score = 0
                        for p in mains:
                            if p[1] > highest_polarity_score:
                                highest_polarity = p[0]
                                highest_polarity_score = p[1]

                        self.stem_polarity_map[search_term] = highest_polarity
                else:
                    self.stem_polarity_map[search_term] = commons[0][0]

                #commons.append([new_words_frequencies_keys[w_i][1], new_words_frequencies_values[w_i]])
                #print(commons)
                #print("concludes to:", self.stem_polarity_map[search_term])


    def getPolarity(self, word):
        try:
            return self.polarity_map[word]
        except KeyError:
            return 0

    def getPolarityOfStem(self, word):
        try:
            return self.stem_polarity_map[word]
        except KeyError:
            return 0

