from deyclassifier import DeyClassifier


class DeyFeatures(DeyClassifier):
    def __init__(self):
        #super(DeyClassifier, self).__init__()
        DeyClassifier.__init__(self)

    def initialise(self, all_content):
        self.initialiseModel(all_content, True)

    def preprocessSingle(self, content, filter_words):
        # Use all words present to determine which are too common
        tokens = []
        unnormalised_tokens = []

        if not isinstance(content, str):
            string_t = content.decode("utf-8").lower()
        else:
            string_t = content

        t = self.normaliseSymbols(string_t)
        token_list = t.split()

        # Get rid of infrequent words and manual stop words
        if filter_words:
            token_list = self.normaliseStopwords(token_list, self.stop_words_p1)
        else:
            token_list = self.normaliseStopwords(token_list, self.stop_words_p2)

        string_t = self.normaliseSlang(token_list).lower()
        token_list = string_t.split()

        tokens.append(token_list)
        unnormalised_tokens.append(t.split())

        stemmed_tokens = self.stem(tokens)

        return tokens, stemmed_tokens, unnormalised_tokens