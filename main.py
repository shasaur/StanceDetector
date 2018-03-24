import codecs

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

import random
import unicodedata
import codecs

from sklearn.naive_bayes import GaussianNB

# Download and split the dataset into different columns
# train['tweet'] = np.char.decode(train['tweet'], encoding='windows-1252')
train_all=np.genfromtxt('C:\\Users\\shasa\\scikit_learn_data\\se16-train.txt', delimiter='\t', comments="((((",
                dtype={'names': ('id', 'target', 'tweet', 'stance'),
                       'formats': ('int', 'S50', 'S200', 'S10')})
train = train_all[train_all['target'] == b'Hillary Clinton']
print(len(train['target']), ' of ', len(train_all['target']))

test_all=np.genfromtxt('C:\\Users\\shasa\\scikit_learn_data\\se16-test-gold.txt', delimiter='\t', comments="((((",
                dtype={'names': ('id', 'target', 'tweet', 'stance'),
                       'formats': ('int', 'S50', 'S200', 'S10')})
test = test_all[test_all['target'] == b'Hillary Clinton']
print(len(test['target']), ' of ', len(test_all['target']))

# Testing tweet content
print(train[0][2])
print(train[0][2].index("I"))

# SVM feature extraction, transformation & training
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)),
])
text_clf.fit(train['tweet'], train['stance'])

#### TESTING THE MODEL ####

# Frequency counting single word baseline
predicted = text_clf.predict(test['tweet'])

# Frequency counting * Inverse document frequency
count_vect = CountVectorizer()#ngram_range=(3, 4), analyzer='char')
X_train_counts = count_vect.fit_transform(train_all['tweet'])
print(X_train_counts.shape)

# Getting relative frequencies of phrases
# -This is the classic of "look at all documents, how unique is this word for this document?"
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts) # Term Frequency * Inverse Doc freq transformer

# Training a classifier.py
clf =SGDClassifier(loss='hinge', penalty='l2',
       alpha=1e-3, random_state=42,
       max_iter=4, tol=None).fit(X_train_tfidf, train_all['stance'])

docs_new = test['tweet']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

## Boolean occurange single word baseline
# count_vect = CountVectorizer() #ngram_range=(1, 3)
# X_train_counts = count_vect.fit_transform(train['tweet'])
#
# # Loop over any non-zero fields in the sparse matrix
# # print(X_train_counts[0][0])
# # print(X_train_counts.getnnz(axis=None))
# # print(X_train_counts.getnnz(axis=0), " , ", len(X_train_counts.getnnz(axis=0)))
# # print(X_train_counts.getnnz(axis=1)[:8], " , ", len(X_train_counts.getnnz(axis=1)))
#
# word_occurances = X_train_counts.getnnz(axis=0)
# vocab_sizes = X_train_counts.getnnz(axis=1)
# # X_train_counts[0] =  [(0, 2165)	1 (0, 1881)	1]
# # print(X_train_counts[0])
# # print("at 1: ", X_train_counts[1])
# # print("at 2: ", X_train_counts[2])
# # print("at 3: ", X_train_counts[3])
# # print("at 4: ", X_train_counts[4])
#
# # Loop over all the tweets
# #for t in range(len(vocab_sizes)):
#     # Loop over all the possible words in the tweet
#     #for w in range(len(word_occurances)):
#         # If the current feature is greater than 0, just set it to 1 to keep a boolean relationship
#         # try:
#         #print(t, ",", w, ",", X_train_counts[t][w])
#         #print(t,",",w,",",X_train_counts[t][w][1])
#         # except IndexError:
#         #     print(t, ",", w, ",e")
#         #if X_train_counts[t][w] != 0:
#         #    X_train_counts[t][w] = 1
#
# #print(X_train_counts)
#
# # Getting relative frequencies of phrases
# # -This is the classic of "look at all documents, how unique is this word for this document?"
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts) # Term Frequency * Inverse Doc freq transformer
#
# print(X_train_tfidf[0].)
# print("at 1: ", X_train_tfidf[1])
# print("at 2: ", X_train_tfidf[2])
# print("at 3: ", X_train_tfidf[3])
# print("at 4: ", X_train_tfidf[4])
#
#
# # Training a classifier.py
# clf =SGDClassifier(loss='hinge', penalty='l2',
#        alpha=1e-3, random_state=42,
#        max_iter=5, tol=None).fit(X_train_tfidf, train['stance'])
#
# docs_new = test['tweet']
# X_new_counts = count_vect.transform(docs_new)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)
#
# predicted = clf.predict(X_new_tfidf)

## Random baseline
# for i in range(len(test['stance'])):
#      predicted[i] = random.choice([b'AGAINST', b'NONE', b'FAVOR'])

## Majority baseline
# for i in range(len(test['stance'])):
#     predicted[i] = b'AGAINST'

print(predicted[0])

# RESULTS
print(predicted)
print(np.mean(predicted == test['stance']))
print(metrics.classification_report(test['stance'], predicted))

# Get datasets and extract one subset from them to train for
# contents = 'C:/Users/shasa/scikit_learn_data/se16-train.txt'
#
# train_all=np.genfromtxt(contents, delimiter='\t', comments="((((",
#                 dtype={'names': ('id', 'target', 'tweet', 'stance'),
#                        'formats': ('int', 'S50', 'S200', 'S10')})
#
# train = train_all[train_all['target'] == b'Hillary Clinton']
# print(len(train['target']), ' of ', len(train_all['target']))
#
# test_all=np.genfromtxt('C:/Users/shasa/scikit_learn_data/se16-test-gold.txt', delimiter='\t', comments="((((",
#                 dtype={'names': ('id', 'target', 'tweet', 'stance'),
#                        'formats': ('int', 'S50', 'S200', 'S10')})
#
# test = test_all[test_all['target'] == b'Hillary Clinton']
# print(len(test['target']), ' of ', len(test_all['target']))
#
#
# # Extract features
# count_vect = CountVectorizer(ngram_range=(1, 3))
# X_train_counts = count_vect.fit_transform(train['tweet'])
#
# print(X_train_counts.shape)



# print(np.loadtxt('C:/Users/shasa/scikit_learn_data/se16-train.txt', delimiter='\t', comments="((((",
#                 skiprows=1, dtype={'names': ('id', 'target', 'tweet', 'stance'),
#                        'formats': ('int', 'S10', 'S200', 'S10')}))   # columns names if no header

# arr = np.fromiter(codecs.open("C:/Users/shasa/scikit_learn_data/se16-train.txt", encoding="utf-8"),
#                   delimiter='\t', comments="((((",
#                   skiprows=1, dtype={'names': ('id', 'target', 'tweet', 'stance'),
#                                      'formats': ('int', 'S10', 'S200', 'S10')})


# vect = TfidfVectorizer()
# X = vect.fit_transform(df['tweets'])
# y = df['class']
#
# # Tokenize text
# count_vect = CountVectorizer(ngram_range=(1, 3))
# X_train_counts = count_vect.fit_transform(twenty_train.data)