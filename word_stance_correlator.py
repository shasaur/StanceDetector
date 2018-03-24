from sentiwordnet_lexicon import SentiWordNetLexicon

class Word:
    def __init__(self, bins, polarity):
        self.bins = bins
        self.polarity = polarity

class WordStanceCorrelater:
    def getBin(self, stance):
        if stance == b'FAVOR':
            return 1
        else:
            return 0

    def __init__(self, tweets, words, stances, mpqa_lexicon=None, swn_lexicon=None):
        self.words = {}
        total_stances = [0,0]

        for w in words:
            if swn_lexicon is not None:
                self.words[w] = Word([0,0], swn_lexicon.getSentimentOfWord(w))
            if mpqa_lexicon is not None:
                self.words[w] = Word([0, 0], mpqa_lexicon.getPolarity(w))

        # Count the presence of each word in the each stance class
        for i in range(len(tweets)):

            unique_tokens = []

            # Get the set of words in the tweeet
            for token in tweets[i]:
                if token not in unique_tokens:
                    unique_tokens.append(token)

            # Add respective stance occurances to each unique word
            for token in unique_tokens:
                #print("adding up", token, self.getBin(stances[i]))
                if token in words:
                    self.words[token].bins[self.getBin(stances[i])] += 1

            total_stances[self.getBin(stances[i])] += 1

        for w in words:
            acc1 = self.words[w].bins[0] / total_stances[0]
            acc2 = self.words[w].bins[1] / total_stances[1]
            acc1_str = "%.3f" % round(acc1, 3)
            acc2_str = "%.3f" % round(acc2, 3)

            total_accuracy = acc1+acc2

            if total_accuracy > 0:
                if self.words[w].polarity == 0:
                    row_print = [w, acc1_str, acc2_str, str(acc1 / total_accuracy)]
                else:
                    row_print = [w, acc1_str, acc2_str, str(acc2 / total_accuracy)]

                print(",".join(row_print))

            #print("results", w, , self.words[w].bins[1]/total_stances[1])