import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import wordnet as wn

from enum import Enum
from sklearn.svm import LinearSVC

from porterstemmer import PorterStemmer
from mpqalexicon import MPQALexicon
from slanglexicon import SlangLexicon
from sentiwordnet_lexicon import SentiWordNetLexicon

from sklearn import metrics

import test_utilities

from popularity_counter import PopCounter
from collections import OrderedDict
from word_stance_correlator import WordStanceCorrelater

class Tweet(Enum):
    id = 0
    target = 1
    content = 2
    stance = 3

class P1F(Enum):
    mpqa = 0
    swn = 1
    adjectives = 2

class P2F(Enum):
    swn = 0
    mpqa = 1
    frame = 2
    target = 3
    wordgrams = 4
    chargrams = 5

class DeyClassifier:
    def __init__(self):
        self.polarity_lexicon = MPQALexicon()
        self.swn_polarity_lexicon = SentiWordNetLexicon()

        self.mpqa_weights = []

        self.slang_lexicon = SlangLexicon()

        self.count_vectoriser = None
        self.word_occurrence_matrix = None
        self.adjective_inv_vocab = None

        self.classifier_p1 = None

        self.test_array = []

        self.stop_words_p1=[]
        self.stop_words_p2=[]
        #self.stop_words_p1 = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'is', 'it', 'its', 'of', 'that', 'the', 'to', 'was', 'were', 'will', 'with', 'because', 'i', 'want', 'american', 'women', 'woman', 'president', 'semst', 'years', 'hillary', 'never', 'she', 'email', 'media', 'take', 'gop', 'vote', 'hillaryclinton', 'hillaryforia', 'world', 'supporting', 'you', 'remember', 'war', 'before', 'whatever', 'benghazi', 'day', 'ever', 'justice', 'have', 'rights', 's', 'clinton', 'would', 'term', 'her', 'emails', 'lies', 'your', 'face', 'bill', 'wants', 'dude', 'if', 'can', 'do', 'what', 'did', 'doing', 'who', 'm', 'didn', 't', 'realize', 'how', 'run', 'office', 'this', 'libertynothillary', 'tcot', 'uniteblue', 'there', 'when', 'think', 'next', 'either', 'or', 'obama', 'hey', 'way', 'votes', 'too', 'says', 'campaign', 'secretary', 'state', 'chrischristie', 'his', 'let', 'are', 'words', 'blame', 'time', 'for', '000', 'our', 'iraq', 'know', 'year', 'could', 'really', 'country', 'make', 'p2', 'why', 'people', 'around', 'hate', 'spent', 'problem', 'not', 'party', 'house', 'against', 'news', 'am', 'person', 'usa', 'isis', 'bush', 'doesn', 'care', 'coming', 'don', 'finally', 'call', 'man', 'they', 'dog', 'thehill', 'me', 'leave', 'focus', 'issues', '2016', 'candidates', 'cruz', 'donors', 'clintons', '1', 'marriage', 'equality', 'readyforhillary', 'needs', '10', 'yes', 'tell', 'us', 'get', 'lying', 'again', 'lie', 'nothing', 'sending', 'love', 'victory', 'marcorubio', 'speech', 'd', 'called', 'nohillary2016', 'whyimnotvotingforhillary', 'democrat', 'their', 'shit', 'gets', 'keep', 'theblaze', 'taking', 'muslim', 'rid', '11', 'also', 'thedemocrats', 'lied', 'agree', 'randpaul', 'americans', 'education', 'family', 'but', 'freedom_justice_equality_education', 'innovation', 'development', 'happy_life', 'utopia', 'idea', 'machine', 'those', 'tweet', 'job', 'answer', 'today', 'happen', 'say', 'rest', 'yawn', 'iowa', 'saudi', 'govt', 'monica', 'politico', 'watch', 'god', 'liberals', 'china', 'works', 'voting', 'talk', 'chelsea', 'end', 'foundation', 'clintoncash', 'foxnews', 'server', 'candidate', 'potus', 'unitedstates', 'took', 'marymorientes', 'anything', 'change', 'longer', 'hrc', 'baltimore', 'faith', 'believe', 'mistakes', 'politics', 'feel', 'wasn', '2013', 'newamericancentury', 'billclinton', 'teen', 'hope', 'texas', 'pay', 'keeps', 'join', 'mention', 'follow', 'of', 'warcraft', '(online', 'gaming)', 'wakeupamerica', 'politician', 'gays', 'hypocrite', 'berniesanders', 'democrats', 'guy', 'until', 'election', 'dem', 'voters', 'team', 'wife', 'vs', 'bernie', 'sanders', 'speak', '5', 'gee', 'makes', 'making', '7', 'got', 'w', 'talking', 'blacks', 'should', 'trust', 'between', 'policy', 'death', 'workers', 'equalityforall', 'futuretxleader', 'n', 'tea', 'been', 'bless', 'bernie2016', 'jstines3', 'mouth', 'night', 'republicans', 'leader', 'weekend', 'l', 'where', 'life', '/', "you're", 'rock', 'line', 'donaldtrump', 'awesome', 'supporter', 'said', 'hilary', 'deleted', 'scandal', 'into', 'county', 'joke', 'story', 'mom', 'slogan', 'truth', 'shirt', 'idiot', 'else', 'biggest', 'ass', 'these', 'lovewins', 'lgbt', 'laws', 'reality', 'knew', 'tv', 'show', 'wait', 'aren', 'pretending', 'stophillary2016', 'gonna', 'lose', 'told', 'cant', 'might', 'fire', 'record', 'c', 'tlot', 'fox', 'cost', 'parent', 'is', 'watching?', 'rubio2016', 'continue', 'univision', 'joebiden', 'putin', 'supporters', 'hill', 'hillaryforsc', 'queen', 'anyone', 'drmartyfox', 'manage', 'hillaryemails', 'fax', 'released', '0', 'freeallfour', 'hillaryinnh', 'killary', '_']

        self.stop_words_p1 = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with', 'i', 'to', 'rubio', 'while', 'community', 'barackobama', 'world', 'you', 'whatever', 'ever', 'long', 'your', 've', 'my', 'm', 'u', 'run', 'this', 'up', 'only', 'words', 'like', 'our', 'year', 'out', 'loud', 'people', 'here', 'even', 'party', 'am', 'jeb', 'don', 'yet', 'finally', 'every', 'me', '2016', 'over', 'had', 'twitter', 'cruz', 'real', 'america', 'name', 'still', 'tedcruz', 'getting', '3', 'randpaul', 'freedom_justice_equality_education', 'innovation', 'development', 'happy_life', 'utopia', 'hide', 'hell', 'use', 'rest', 'meaning', 'yawn', 'trump', 'pretty', 'white', 'already', 'very', 'corrupt', 'marymorientes', 'change', 'baltimoreriots', 'raised', 'thing', 'hear', 'of', 'warcraft', '(online', 'gaming)', 'voters', 'destroy', 'bernie', 'sanders', 'always', 'poor', '5', 'realdonaldtrump', 'making', 'please', 'trust', 'presidential', 'policy', 'thanks', 'workers', 'big', 'lead', 'liar', 'jebbush', 'county', 'excited', 'fellowssc', 'hug', 'bad', 'truth', 'guy', 'in', 'life', 'scotus', 'wins', 'lovewins', 'true', 'must', 'enough', 'broke', 'having', 'fire', 'feelthebern', 'tonight', 'flip', 'issue', 'ecstasy', 'transparency', 'chance', 'less', 'million', 'control', 'zero']
        #self.stop_words_p2 = ['i', 'to', 'rubio', 'while', 'community', 'barackobama', 'world', 'you', 'whatever', 'ever', 'long', 'your', 've', 'my', 'm', 'u', 'run', 'this', 'up', 'only', 'words', 'like', 'our', 'year', 'out', 'loud', 'people', 'here', 'even', 'party', 'am', 'jeb', 'don', 'yet', 'finally', 'every', 'me', '2016', 'over', 'had', 'twitter', 'cruz', 'real', 'america', 'name', 'still', 'tedcruz', 'getting', '3', 'randpaul', 'freedom_justice_equality_education', 'innovation', 'development', 'happy_life', 'utopia', 'hide', 'hell', 'use', 'rest', 'meaning', 'yawn', 'trump', 'pretty', 'white', 'already', 'very', 'corrupt', 'marymorientes', 'change', 'baltimoreriots', 'raised', 'thing', 'hear', 'of', 'warcraft', '(online', 'gaming)', 'voters', 'destroy', 'bernie', 'sanders', 'always', 'poor', '5', 'realdonaldtrump', 'making', 'please', 'trust', 'presidential', 'policy', 'thanks', 'workers', 'big', 'lead', 'liar', 'jebbush', 'county', 'excited', 'fellowssc', 'hug', 'bad', 'truth', 'guy', 'in', 'life', 'scotus', 'wins', 'lovewins', 'true', 'must', 'enough', 'broke', 'having', 'fire', 'feelthebern', 'tonight', 'flip', 'issue', 'ecstasy', 'transparency', 'chance', 'less', 'million', 'control', 'zero']


        self.p1f_active = {}
        self.p2f_active = {}
        self.togglePhaseOneFeatures()
        self.togglePhaseTwoFeatures()

        self.triclass_mode = False

    def togglePhaseOneFeatures(self, swn_score=True, mpqa_score=True, adjective_occurance=True):
        self.p1f_active[P1F.mpqa] = mpqa_score
        self.p1f_active[P1F.swn] = swn_score
        self.p1f_active[P1F.adjectives] = adjective_occurance

    def togglePhaseTwoFeatures(self, swn_score=True, mpqa_score=True, frame_semantics=True, target_detection=True,
                               word_ngrams=True, char_ngrams=True):
        self.p2f_active[P2F.swn] = swn_score
        self.p2f_active[P2F.mpqa] = mpqa_score
        self.p2f_active[P2F.frame] = frame_semantics
        self.p2f_active[P2F.target] = target_detection
        self.p2f_active[P2F.wordgrams] = word_ngrams
        self.p2f_active[P2F.chargrams] = char_ngrams

    def toggleTriclassMode(self):
        self.triclass_mode = True

    ## HELPER FUNCTIONS
    def normaliseSymbols(self, content):
        content = re.sub(r'[^\w]', ' ', content)
        return content
        # simplified = content.replace(':', '')
        # simplified = simplified.replace('"', '')
        # simplified = simplified.replace("'", "")
        # simplified = simplified.replace(".", "")
        # simplified = simplified.replace(",", "")
        # simplified = simplified.replace("!", "")
        # return simplified

    def normaliseStopwords(self, tokens, stop_words):
        # Get only the words that appear in the collections of tweets
        # indices = self.word_occurrence_matrix.nonzero()
        # no_of_words = len(indices[0])
        #
        # tweet_tokens = []
        # w = 0

        new_tokens = []
        for token in tokens:
            if token not in stop_words:
                new_tokens.append(token)

        # tweet = ' '.join(tokens)
        # tokenized = self.count_vectoriser.fit_transform(sentence)
        # result = self.count_vectoriser.inverse_transform(tokenized)

        return new_tokens

        # return new array

        # for t in range(no_of_tweets):
        #     words = []
        #
        #     # indices[0] = (0, 1), (0, 3), (0,10), (1, 2), (1, 10), ... , (551, 15)
        #
        #     # basically go through the sparse matrix and add that word in the tweet
        #     # if it exists in the matrix (i.e. after filtering, add all of the words left)
        #     while w < no_of_words and indices[0][w] == t:  # trust
        #
        #         # entry_index = counts_sparse_csr_matrix[t].getcol(w).argmax()
        #         # sentence.append(count_vect.get_feature_names()[entry_index])
        #
        #         current_token = count_vectoriser.get_feature_names()[indices[1][w]]
        #         words.append(current_token)
        #         w += 1
        #
        #     tweet_tokens.append(words)


    ## HELPER FUNCTIONS FIRST PHASE TRAINING
    def isAdjective(self, word):
        synonyms = wn.synsets(word)

        adjective = False
        for s in range(len(synonyms)):
            if synonyms[s].lemma_names()[0] == word and (synonyms[s].pos() == 'a' or synonyms[s].pos() == 's'):
                adjective = True

        return adjective

        # synonyms = wn.synsets(word)
        # if len(synonyms) != 0:
        #     print(word, "with synonym", synonyms[0],  " has pos:", synonyms[0].pos())
        #     tmp = synonyms[0].pos()
        #     if tmp == 'a':
        #         return True
        #     else:
        #         return False
        # else:
        #     return False

    def getAdjectives(self, words):
        adjectives = []

        for w in words:
            if self.isAdjective(w):
                adjectives.append(w)

        return adjectives

    def getAdjectiveOccurance(self, tokens, adj_indices):
        adjective_occurrence = []

        # Initialise feature vector as no occurances across the board
        for w in range(len(adj_indices.keys())):
            adjective_occurrence.append(False)

        # Fill in the occurrences only based on tweet tokens
        for w in tokens:
            if w in adj_indices.keys():
                adjective_occurrence[adj_indices[w]] = True

            # DOES THE ADJECTIVE KEY SET CONTAIN THIS WORD?
            # IF SO, CHANGE ITS ENTRY IN THE ADJECTIVE_OCCURANCE ARRAY AT POSITION ADJECTIVES.GET_VALUE(WORD)

        return adjective_occurrence

    def calculate_mpqa(self, words, are_stemmed, print_debug):
        total = 0
        test_array = []
        for w in words:
            if not are_stemmed:
                total += self.polarity_lexicon.getPolarity(w)
                test_array.append([w, self.polarity_lexicon.getPolarity(w)])
            else:
                total += self.polarity_lexicon.getPolarityOfStem(w)
                test_array.append([w, self.polarity_lexicon.getPolarity(w)])

        if print_debug:
            if are_stemmed:
                print('normal:', test_array)
            else:
                print('stemmed:', test_array)

        return total

    def calculate_swn_score(self, words, print_debug):
        total = 0
        test_array = []
        for w in words:
            total += self.swn_polarity_lexicon.getSentimentOfWord(w)
            test_array.append([w, self.swn_polarity_lexicon.getSentimentOfWord(w)])

        if print_debug:
            print(test_array)

        return total

    ## HELPER FUNCTIONS 2ND PHASE TRAINING
    def addOccurancesToCount(self, existing_grams, occurrences):
        for o in occurrences:
            if o in existing_grams.keys():
                existing_grams[o] += 1
            else:
                existing_grams[o] = 1

    def getCharNGramFrequencies(self, tokens):
        all_grams = {}

        # char_trigrams = set()
        # char_4grams = set()

        for t in tokens:
            char_grams = set()

            for w in t:
                current_grams = [w[i:i + 2] for i in range(len(w) - 1)]
                self.addOccurancesToCount(all_grams, current_grams)

                current_grams = [w[i:i + 3] for i in range(len(w) - 2)]
                self.addOccurancesToCount(all_grams, current_grams)

                current_grams = [w[i:i + 4] for i in range(len(w) - 3)]
                self.addOccurancesToCount(all_grams, current_grams)

        return all_grams

    ## PREPROCESSING
    # Returns the tweet split into tokens
    # def tokenise(self, tweet):
    #     tweet_string = tweet.decode("utf-8")
    #     return self.split_poorly(tweet_string)

    def removeSymbols(self, tokens):
        new_tokens = []
        for t in tokens:
            new_tokens.append(self.simplify(t))

        return new_tokens

    def normaliseSlang(self, tokens):
        for i in range(len(tokens)):
            if tokens[i].upper() in self.slang_lexicon.getSlang():
                # Convert acronym to full set of terms
                tokens[i] = str(self.slang_lexicon.getMeaning(tokens[i]))

        # <convert to Han-Baldwin norm>
        # <back to sentence string>

        return ' '.join(tokens)

    def setVocabulary(self, tweets, isPhase1):
        self.count_vectoriser = CountVectorizer()
        self.word_occurrence_matrix = self.count_vectoriser.fit_transform(tweets)
        all_words = self.count_vectoriser.get_feature_names()

        if isPhase1:
            self.count_vectoriser = CountVectorizer(min_df=0.004,#0.004 was found best at phase 2 optimum
                                                    stop_words=self.stop_words_p1)
        else:
            self.count_vectoriser = CountVectorizer(min_df=0.004, # 0.004 was found best at phase 2 optimum
                                                    stop_words=self.stop_words_p2)

        self.word_occurrence_matrix = self.count_vectoriser.fit_transform(tweets)  # (i,j) = value at pos ij
        kept_words = self.count_vectoriser.get_feature_names()

        if isPhase1:
            self.stop_words_p1.extend(set(all_words).difference(set(kept_words)))
        else:
            self.stop_words_p2.extend(set(all_words).difference(set(kept_words)))

    def setAdjVocabulary(self, tweets):
        self.adj_count_vectoriser = CountVectorizer(min_df=0.004)
        self.adj_count_vectoriser.fit_transform(tweets)

    def setNGramVocab(self, tweets):
        self.ngrams = self.count_vectoriser.get_feature_names()

    def setNCharGramVocab(self, tokens):
        # get total frequency count of each ngram for all documents
        self.ngrams_frequency_map = self.getCharNGramFrequencies(tokens)
        self.character_ngrams = set()

        min_df = 0.01

        total_frequency = 0
        for ngram_frequency in self.ngrams_frequency_map.values():
            total_frequency += ngram_frequency

        # document wide thresholding
        for ngram in self.ngrams_frequency_map.keys():
            if float(self.ngrams_frequency_map[ngram])/float(total_frequency) >= min_df:
                self.character_ngrams.add(ngram)

        print(self.character_ngrams)

    def stem(self, tweets):
        # stemmed_count_vect = CountVectorizer(max_df=0.5, min_df=0.01)  # ngram_range=(1, 3)
        # stemmed_counts_sparse_csr_matrix = stemmed_count_vect.fit_transform(
        #     tweets)  # (i,j) = value at pos ij
        # stemmed_tweet_tokens = getNewTweetsInTokens(stemmed_counts_sparse_csr_matrix, stemmed_count_vect)

        stemmed_tweet_tokens = []
        stemmer = PorterStemmer()

        for t in range(len(tweets)):
            #tweet_tokens = self.tokenise(tweets[t])
            tweet_tokens = tweets[t]

            new_words = ["" for x in range(len(tweet_tokens))]
            for w in range(len(tweet_tokens)):
                new_words[w] = stemmer.stem(tweet_tokens[w], 0, len(tweet_tokens[w]) - 1)

            # Store results
            stemmed_tweet_tokens.append(new_words)

        return stemmed_tweet_tokens

    ## FIRST PHASE TRAINING
    def setFeatures1(self, tokens, stemmed_tokens, full_tokens):
        print()
        print("--------------------------- CALCULATING PHASE 1 FEATURES ---------------------------")

        features = [[] for j in range(len(tokens))]

        self.test_array = []

        # Workout the MPQA combined score of tweek tokens and stemmed tokens
        if (self.p1f_active[P1F.mpqa]):
            for t in range(len(tokens)):
                # print()
                # print('normal:')
                mpqa_of_tweet = self.calculate_mpqa(tokens[t], False, False)
                mpqa_of_tweet += self.calculate_mpqa(stemmed_tokens[t], False, False)

                if mpqa_of_tweet > 2 or mpqa_of_tweet < -2:
                    features[t].append(True)
                else:
                    features[t].append(False)

                # features[t].append(mpqa_of_tweet)

                self.test_array.append(mpqa_of_tweet)


        if self.p1f_active[P1F.swn]:
            for t in range(len(tokens)):
                swn_score_of_tweet = self.calculate_swn_score(tokens[t], False)
                features[t].append(swn_score_of_tweet)


        # Computing features based on model adjectives
        if (self.p1f_active[P1F.adjectives]):

            for t in range(len(tokens)):
                boolean_adjective_features = self.getAdjectiveOccurance(tokens[t], self.adjective_inv_vocab)

                # go through each boolean feature and add it to the feature list for that tweet
                features[t].extend(boolean_adjective_features)

        return features

    ## SECOND PHASE TRAINING
    # Find appearances of preprocessed ngrams in tweets and build their feature vector
    def getCharNGrams(self, tokens, features):

        for t in range(len(tokens)):
            current_grams = set()

            for w in tokens[t]:
                current_grams = current_grams.union(set([w[i:i + 2] for i in range(len(w) - 1)]))
                current_grams = current_grams.union(set([w[i:i + 3] for i in range(len(w) - 2)]))
                current_grams = current_grams.union(set([w[i:i + 4] for i in range(len(w) - 3)]))

            #print(current_grams)

            # For any defined character_ngrams that appear in this tweet, toggle their feature
            for f in self.character_ngrams:
                if f in current_grams:
                    #print(f)
                    features[t].append(True)
                else:
                    features[t].append(False)

        return features

    def getDocumentFrequencies(self, frequencies):
        document_frequency = np.array([0] * len(frequencies[0]))
        for i in range(len(frequencies)):
            document_frequency += np.array(frequencies[i])
        return document_frequency

    def tweetContains(self, word, words):
        contains_target = False
        for w in words:
            if w == word:
                contains_target = True
        return contains_target

    def setFeatures2(self, raw_tweets, tokens, stemmed_tokens):
        print()
        print("--------------------------- CALCULATING PHASE 2 FEATURES ---------------------------")

        features = [[] for j in range(len(tokens))]

        ## SENTIMENT CLASSIFICATION
        # # Initialise adjectives
        # adjective_vocab = self.getAdjectives(self.count_vectoriser.get_feature_names())
        #
        # # Initialise indexes for referencing adjectives
        # adjective_inv_vocab = {adjective_vocab[i]: i for i in range(len(adjective_vocab))}

        # SENTIMENT SCORES
        if self.p2f_active[P2F.mpqa]:
            for t in range(len(tokens)):
                mpqa_of_tweet = self.calculate_mpqa(tokens[t], False, False)
                mpqa_of_tweet += self.calculate_mpqa(stemmed_tokens[t], False, False)

                features[t].append(mpqa_of_tweet)
        if self.p2f_active[P2F.swn]:
            for t in range(len(tokens)):
                #print(raw_tweets[t])
                swn_score_of_tweet = self.calculate_swn_score(tokens[t], False)
                #print()
                features[t].append(swn_score_of_tweet)

        # FRAME SEMANTICS
        if self.p2f_active[P2F.frame]:
            for t in range(len(tokens)):
                features[t].append(0)

        # TARGET DETECTION
        if self.p2f_active[P2F.target]:
            for t in range(len(tokens)):
                if not isinstance(raw_tweets[t], str):
                    sss = raw_tweets[t].decode('UTF-8').lower()
                else:
                    sss = raw_tweets[t].lower()

                features[t].append('hillary clinton' in sss)
                # features[t].append(self.tweetContains('hillaryclinton', tokens[t]))
                # features[t].append(self.tweetContains('hillary', tokens[t]))
                # features[t].append(self.tweetContains('clinton', tokens[t]))
                # features[t].append(self.tweetContains('clinton', tokens[t]) and self.tweetContains('hillary', tokens[t]))

        # WORD N-GRAMS
        if self.p2f_active[P2F.wordgrams]:
            print("word n-grams", len(features), len(self.ngrams))
            print("word n-grams", features[0])
            print("word n-grams", self.ngrams[0])

            for t in range(len(tokens)):
                ngram_occurence = [False] * len(self.ngrams)
                for token in tokens[t]:
                    if token in self.ngrams:
                        ngram_occurence[self.ngrams.index(token)] = True

                features[t] = features[t] + ngram_occurence

        for i in range(len(features)):
            features[i].append(0)

        # CHARACTER N-GRAMS
        if self.p2f_active[P2F.chargrams]:
            features = self.getCharNGrams(tokens, features)

        # # # didn't work anyway because above wasn't actaully counting, just setting to 1
        # # shit threshold
        # for t in range(no_of_tweets):
        #     for c in range(len(self.character_ngrams)):
        #         index = c + previous_feature_length
        #         if features[t][index] > 1:
        #             features[t][index] = 1
        #         else:
        #             features[t][index] = 0

        # total_frequencies = [0 for i in range(len(self.character_ngrams))]
        # for t in range(no_of_tweets):
        #     for c in range(len(self.character_ngrams)):
        #         index = c + previous_feature_length
        #         total_frequencies += features[t][index]

        # print("**First feature vector")
        # print(tokens[0])
        # print(features[0])
        # print('**')

        return features

    ## TEMPORARY LOOKUP, REPLACE WITH HASHMAP PLS WHEN POSSIBLE
    def lookupWordNGram(self, wordGram):
        for i in range(len(self.character_ngrams)):
            if self.character_ngrams[i] == wordGram:
                return i

    ## CONVERT STANCE LABELS TO A LINEAR SEPERATION PROBLEM
    def getFirstPhaseStances(self, stances):
        first_phase_stances = []
        for i in range(len(stances)):
            first_phase_stances.append(b'OTHER' if stances[i] == b'AGAINST' or stances[i] == b'FAVOR'
                                       else stances[i])

        return first_phase_stances

    def getSecondPhaseStances(self, stances):
        new_stances = []
        for i in range(len(stances)):
            if stances[i] == b'AGAINST' or stances[i] == b'FAVOR':
                new_stances.append(stances[i])

        return new_stances

    # def getSecondPhaseTweets(self, tweets, stances):
    #     new_tweets = []
    #     for i in range(len(stances)):
    #         if stances[i] == b'AGAINST' or stances[i] == b'FAVOR':
    #             new_tweets.append(tweets[i])
    #
    #     return new_tweets

    def initialiseModel(self, tweets, isPhase1Model):
        print()
        print("----------------------------------- INITIALISING -----------------------------------")
        self.setVocabulary(tweets, isPhase1Model)

        if isPhase1Model:
            self.setAdjVocabulary(tweets)
        else:
            self.setNGramVocab(tweets)

    def preprocess(self, tweets, isPhase1, stances=[]):
        print()
        print("----------------------------------- PREPROCESSING -----------------------------------")

        # Use all words present to determine which are too common
        tokens = []
        unnormalised_tokens = []
        for t in tweets:
            if not isinstance(t, str):
                string_t = t.decode("utf-8").lower()
            else:
                string_t = t
            t = self.normaliseSymbols(string_t)
            token_list = t.split()

            # Get rid of infrequent words and manual stop words
            if isPhase1:
                token_list = self.normaliseStopwords(token_list, self.stop_words_p1)
            else:
                token_list = self.normaliseStopwords(token_list, self.stop_words_p2)

            string_t = self.normaliseSlang(token_list).lower()
            token_list = string_t.split()

            tokens.append(token_list)
            unnormalised_tokens.append(t.split())

        #print("too l8 m8")

        stemmed_tokens = self.stem(tokens)

        # print('my stopwords:', self.stop_words)
        # print('all stopwords', self.count_vectoriser.get_stop_words())

        return tokens, stemmed_tokens, unnormalised_tokens

    def defineModelParams(self, tokens, isPhase1):
        print()
        print("------------------------------ DEFINING MODEL PARAMS ------------------------------")

        if isPhase1:
            # Initialise adjectives
            adjective_vocab = self.getAdjectives(self.adj_count_vectoriser.get_feature_names())

            # Initialise indexes for referencing adjectives
            self.adjective_inv_vocab = {adjective_vocab[i]: i for i in range(len(adjective_vocab))}
        else:
            self.setNCharGramVocab(tokens)

    def filterNeutral(self, tweets, tokens, stemmed_tokens, predictions, outputs):
        new_tokens = []
        new_stemmed_tokens = []
        new_stances = []
        new_tweets = []

        for i in range(len(predictions)):
            if not predictions[i] == b'NONE':
                new_tokens.append(tokens[i])
                new_stemmed_tokens.append(stemmed_tokens[i])
                new_stances.append(outputs[i])
                new_tweets.append(tweets[i])

        return new_tweets, new_tokens, new_stemmed_tokens, new_stances

    def printTopKFeatures(self, features, k):
        for i in range(k):
            print(features[i])

    def train(self, train_set):
        tweets = train_set['tweet']
        stances = train_set['stance']

        if self.triclass_mode:
            p1_stances = stances
        else:
            p1_stances = self.getFirstPhaseStances(stances)

        # for i in range(0, 10):
        #     print(tweets[i])

        # Initialise training
        # debug_tweets = self.debug_getSecondPhaseTweets(tweets, stances)
        # debug_tweets = [tweets[i] for i in range(stances == b'FAVOR' or stances==b'AGAINST')]
        # for i in range(len(debug_tweets)):
        #     tweets[i] = self.normalise(debug_tweets[i])

        self.initialiseModel(tweets, True)
        pre1_tokens, pre1_stemmed_tokens, pre1_unnormalised_tokens = self.preprocess(tweets, True)
        self.defineModelParams(pre1_tokens, True)

        fv1 = self.setFeatures1(pre1_tokens, pre1_stemmed_tokens, pre1_unnormalised_tokens)

        self.initialiseModel(tweets, False)
        pre2_tokens, pre2_stemmed_tokens, pre2_unnormalised_tokens = self.preprocess(tweets, False)
        self.defineModelParams(pre2_tokens, False)

        p2_tweets, p2_tokens, p2_stemmed_tokens, p2_stances = self.filterNeutral(tweets, pre2_tokens, pre2_stemmed_tokens, stances, stances)
        fv2 = self.setFeatures2(p2_tweets, p2_tokens, p2_stemmed_tokens)

        #wsc = WordStanceCorrelater(p2_tokens, pc.frequency_map.keys(), p2_stances, swn_lexicon=self.swn_polarity_lexicon)


        print('lengths', len(fv2), len(p2_tokens), len(p2_stances))

        if self.p1f_active[P1F.mpqa] or self.p1f_active[P1F.swn] or self.p1f_active[P1F.adjectives]:
            print()
            print("--------------------------------- TRAINING PHASE 1 ---------------------------------")
            self.printTopKFeatures(fv1, 5)

            #test_utilities.saveFeaturesToWeka('feats_train_p1', '{NONE,OTHER}', fv1, p1_stances)

            self.classifier_p1 = LinearSVC()
            self.classifier_p1 = self.classifier_p1.fit(fv1, p1_stances)

        if (self.p2f_active[P2F.swn] or self.p2f_active[P2F.mpqa] or self.p2f_active[P2F.frame] or
                self.p2f_active[P2F.target] or self.p2f_active[P2F.chargrams] or self.p2f_active[P2F.wordgrams]):
                print()
                print("--------------------------------- TRAINING PHASE 2 ---------------------------------")

               # test_utilities.saveFeaturesToWeka('features_train_p2', '{FAVOR,AGAINST}', fv2, p2_stances)

                self.printTopKFeatures(fv2, 10)

                self.classifier_p2 = LinearSVC()
                self.classifier_p2 = self.classifier_p2.fit(fv2, p2_stances)

        # test_utilities.saveFeaturesToCSV('train', fv1, first_phase_stances, first_phase_stances)

    def quickCount(self, predicted, stances):
        neutral = [0,0] #[incorrect, correct]
        polarity = [0,0] #[incorrect, correct]

        for i in range(len(predicted)):
            if stances[i] == b'NONE':
                if predicted[i] == b'NONE':
                    neutral[1] += 1
                else:
                    neutral[0] += 1
            else:
                if predicted[i] == b'NONE':
                    polarity[0] += 1
                else:
                    polarity[1] += 1

        print('neutral', neutral)
        print('polarity', polarity)

    def combineResults(self, subjectivity_predictions, stance_predictions):
        stance_i = 0
        for i in range(len(subjectivity_predictions)):
            if not subjectivity_predictions[i] == b'NONE':
                subjectivity_predictions[i] = stance_predictions[stance_i]
                stance_i += 1

        return subjectivity_predictions

    ## TESTING
    def test(self, test_set):
        tweets = test_set['tweet']
        stances = test_set['stance']

        if self.triclass_mode:
            p1_stances = stances
        else:
            p1_stances = self.getFirstPhaseStances(stances)

        tokens, stemmed_tokens, unnormalised_tokens = self.preprocess(tweets, True)

        ## Make sure to toggle tweets <-> debug_tweets
        fv1 = self.setFeatures1(tokens, stemmed_tokens, unnormalised_tokens)

        print()
        print("---------------------------------- TESTING PHASE 1 --------------------------------")
        self.printTopKFeatures(fv1, 5)

        predicted_1 = self.classifier_p1.predict(fv1)

        self.quickCount(predicted_1, p1_stances)
        print(np.mean(predicted_1 == p1_stances))
        print(metrics.classification_report(p1_stances, predicted_1))

        ## Save to CSV
        #test_utilities.saveFeaturesToCSV('test', fv1, predicted_1, p1_stances)
        #test_utilities.saveFeaturesToWeka('features_test_p1', '{OTHER,NONE}', fv1, p1_stances)

        print()
        print("---------------------------------- TESTING PHASE 2 --------------------------------")
        tokens, stemmed_tokens, unnormalised_tokens = self.preprocess(tweets, False)
        # Use the second model to predict the polarity of non-neutral tweets
        p2_tweets, p2_tokens, p2_stemmed_tokens, p2_stances = self.filterNeutral(tweets, tokens, stemmed_tokens, predicted_1, stances)
        fv2 = self.setFeatures2(p2_tweets, p2_tokens, p2_stemmed_tokens)

        self.printTopKFeatures(fv2, 5)

        predicted_2 = self.classifier_p2.predict(fv2)

        self.quickCount(predicted_2, p2_stances)
        print(np.mean(predicted_2 == p2_stances))
        print(metrics.classification_report(p2_stances, predicted_2))

        print()
        print("-------------------------------------- RESULTS ------------------------------------")
        # Now join the results for a combined final score
        final_predictions = self.combineResults(predicted_1, predicted_2)

        self.quickCount(final_predictions, stances)
        print(np.mean(final_predictions == stances))
        print(metrics.classification_report(stances, final_predictions))


        # for i in range(len(predicted)):
        #     print('P[',predicted[i], ']  S[', stances[i], '] is', fv1[i], 'with MPQA:', self.test_array[i])

        # predicted = self.classifier_p2.predict(fv2)
        #
        # # print(predicted)
        # print(np.mean(predicted == second_phase_stances))
        # print(metrics.classification_report(second_phase_stances, predicted))


        print()
        print("--------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------")

    def testPhase2(self, test_set):
        tweets = test_set['tweet']
        stances = test_set['stance']

        if self.triclass_mode:
            p1_stances = stances
        else:
            p1_stances = self.getFirstPhaseStances(stances)

        tokens, stemmed_tokens = self.preprocess(tweets)

        print()
        print("---------------------------------- TESTING PHASE 2 --------------------------------")
        # Use the second model to predict the polarity of non-neutral tweets
        p2_tweets, p2_tokens, p2_stemmed_tokens, p2_stances = self.filterNeutral(tweets, tokens, stemmed_tokens, stances, stances)
        fv2 = self.setFeatures2(p2_tweets, p2_tokens, p2_stemmed_tokens)

        self.printTopKFeatures(fv2, 5)

        predicted_2 = self.classifier_p2.predict(fv2)

        print()
        print("-------------------------------------- RESULTS ------------------------------------")
        self.quickCount(predicted_2, p2_stances)
        print(np.mean(predicted_2 == p2_stances))
        print(metrics.classification_report(p2_stances, predicted_2))

        print()
        print("--------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------")

    def testMajorityBaseline(self, test_set):
        tweets = test_set['tweet']
        stances = test_set['stance']

        fv = []
        final_predictions = []
        for i in range(len(tweets)):
            fv.append([True])
            final_predictions.append(b'AGAINST')

        #test_utilities.saveFeaturesToWeka('majority_test', fv, stances)

        print(np.mean(final_predictions == stances))
        print(metrics.classification_report(stances, final_predictions))

    def classify(self, tweets):
        tokens, stemmed_tokens, unnormalised_tokens = self.preprocess(tweets, True)

        ## Make sure to toggle tweets <-> debug_tweets
        fv1 = self.setFeatures1(tokens, stemmed_tokens, unnormalised_tokens)

        print()
        print("---------------------------------- TESTING PHASE 1 --------------------------------")
        self.printTopKFeatures(fv1, 5)

        predicted_1 = self.classifier_p1.predict(fv1)

        ## Save to CSV
        #test_utilities.saveFeaturesToCSV('test', fv1, predicted_1, p1_stances)
        #test_utilities.saveFeaturesToWeka('features_test_p1', '{OTHER,NONE}', fv1, p1_stances)

        print()
        print("---------------------------------- TESTING PHASE 2 --------------------------------")
        tokens, stemmed_tokens, unnormalised_tokens = self.preprocess(tweets, False)

        # Use the second model to predict the polarity of non-neutral tweets
        p2_tweets, p2_tokens, p2_stemmed_tokens, p2_stances = \
            self.filterNeutral(tweets, tokens, stemmed_tokens, predicted_1, predicted_1)
        fv2 = self.setFeatures2(p2_tweets, p2_tokens, p2_stemmed_tokens)

        predicted_2 = self.classifier_p2.predict(fv2)

        print()
        print("-------------------------------------- RESULTS ------------------------------------")
        # Now join the results for a combined final score
        final_predictions = self.combineResults(predicted_1, predicted_2)

        print()
        print("--------------------------------------------------------------------------------")

        return final_predictions
