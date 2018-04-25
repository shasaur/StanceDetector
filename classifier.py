import numpy as np
import nltk
from deyclassifier import DeyClassifier

# def get_nonzero_indices(sparse_array):
#     for i in range(0, sparse_array.get)

print()
print("-- INITIALISING --")
# Download
nltk.download('wordnet')

# Load data
train_set=np.genfromtxt('data\\amplified-training-data.txt', delimiter='\t', comments="((((",
                dtype={'names': ('id', 'target', 'tweet', 'stance'),
                       'formats': ('S20', 'S50', 'S200', 'S10')})
# train_set=np.genfromtxt('C:\\Users\\shasa\\PycharmProjects\\scikit-tutorial\\data\\manual-train.txt', delimiter='\t', comments="((((",
#                 dtype={'names': ('id', 'target', 'tweet', 'stance'),
#                        'formats': ('int', 'S50', 'S200', 'S10')})
hillary_train_set = train_set[train_set['target'] == b'Hillary Clinton']
#hillary_train_set = np.append(hillary_train_set, train_set[train_set['target'] == b'Hillary Clinton'])


test_all=np.genfromtxt('data\\dataset-test.csv', delimiter=',', comments="((((",
                dtype={'names': ('id', 'user', 'target', 'stance', 'tweet'),
                       'formats': ('S10', 'S50', 'S20', 'S10', 'S200')})
# test_all=np.genfromtxt('data\\se16-test-gold.txt', delimiter='\t', comments="((((",
#                 dtype={'names': ('id', 'target', 'tweet', 'stance'),
#                        'formats': ('int', 'S50', 'S200', 'S10')})
hillary_test_set = test_all[test_all['target'] == b'Hillary Clinton']



# Initialise
classifier = DeyClassifier("linear")
classifier.togglePhaseOneFeatures(mpqa_score=True, swn_score=False, adjective_occurance=True)
classifier.togglePhaseTwoFeatures(swn_score=False, mpqa_score=False, frame_semantics=False, target_detection=False,
                                  word_ngrams=True, char_ngrams=True)
# classifier.toggleTriclassMode() # has to be on until algo learns to filter properly

# Run model on data
classifier.train(hillary_train_set)
classifier.test(hillary_test_set)

# classifier.testMajorityBaseline(np.append(hillary_train_set, hillary_test_set))
# classifier.testMajorityBaseline(hillary_test_set)