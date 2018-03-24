from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn


class SentiWordNetLexicon:
    def __getSynonymForms__(self, word):
        synonyms = wn.synsets(word)
        # print(synonyms)

        # Find the synonym form of the word
        forms = []
        for s in range(len(synonyms)):
            if synonyms[s].lemma_names()[0] == word:
                forms.append(synonyms[s])

        return forms

    def getSentimentOfWord(self, word):
        ss_forms = self.__getSynonymForms__(word)

        total_sentiment = 0
        for ss_form in ss_forms:
            sentiment_desc = swn.senti_synset(ss_form.name())
            #print(sentiment_desc)

            total_sentiment += sentiment_desc.pos_score()
            total_sentiment -= sentiment_desc.neg_score()

        if total_sentiment > 0:
            total_sentiment = 1
        elif total_sentiment < 0:
            total_sentiment = -1

        return int(total_sentiment)

    def SOLID_getSentimentOfWord(self, word):
        ss_forms = self.__getSynonymForms__(word)

        total_sentiment = 0
        for ss_form in ss_forms:
            sentiment_desc = swn.senti_synset(ss_form.name())
            #print(sentiment_desc)

            if (sentiment_desc.pos_score() > sentiment_desc.neg_score()):
                sentiment = +1
            elif (sentiment_desc.pos_score() < sentiment_desc.neg_score()):
                sentiment = -1
            else:
                sentiment = 0

            # print(sentiment_desc, sentiment_desc.pos_score(), sentiment_desc.neg_score(), sentiment_desc.obj_score(),
            #       sentiment)

            total_sentiment += sentiment

        if total_sentiment > 1:
            total_sentiment = 1
        elif total_sentiment < -1:
            total_sentiment = -1

        return total_sentiment
