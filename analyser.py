import numpy as np
import nltk
from deyclassifier import DeyClassifier

import csv
from datetime import datetime

print()
print("-- INITIALISING --")
# Download
nltk.download('wordnet')

# Load training data
train_set=np.genfromtxt('C:\\Users\\shasa\\scikit_learn_data\\se16-train.txt', delimiter='\t', comments="((((",
                dtype={'names': ('id', 'target', 'tweet', 'stance'),
                       'formats': ('int', 'S50', 'S200', 'S10')})
hillary_train_set = train_set[train_set['target'] == b'Hillary Clinton']


# Initialise model
classifier = DeyClassifier()
classifier.togglePhaseOneFeatures(mpqa_score=True, swn_score=False, adjective_occurance=True)
classifier.togglePhaseTwoFeatures(swn_score=False, mpqa_score=False, frame_semantics=False, target_detection=False,
                                  word_ngrams=True, char_ngrams=True)

# Prepare model with SemEval16 stance dataset
classifier.train(hillary_train_set)

# Load US election data
print("START@", datetime.now())
dates = []
tweets = []
stances = []
users = []
with open('data\\tweets\\0-58_part1\\tweet_subset_1_part1.csv', 'r', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    i = 0
    for line in reader:
        # Avoid header
        if i!=0:

            if i <= 2000000:
                users.append(line[1])

                date_str = line[2]
                date_str = ' '.join([date_str[0:4], date_str[5:7], date_str[8:10], date_str[11:13]])
                dates.append(datetime.strptime(date_str, '%Y %m %d %H'))

                tweets.append(line[3])

                i += 1
            else:
                break

        else:
            i += 1

print("LOADED@", datetime.now())

hillary_tweets = []
hillary_dates = []
hillary_users = []
for i in range(len(tweets)):
    if "hillary" in tweets[i] or "clinton" in tweets[i]:
        hillary_tweets.append(tweets[i])
        hillary_dates.append(dates[i])
        hillary_users.append(users[i])

print("FILTER@", datetime.now())
print("Tweets including the target:", len(hillary_tweets))

# Analyse election data
hillary_stances = []
hillary_stances = classifier.classify(hillary_tweets)
unique, counts = np.unique(hillary_stances, return_counts=True)
d = dict(zip(unique, counts))
print(len(hillary_stances))
print("FOR:", d[b'FAVOR'])
print("AGAINST:", d[b'AGAINST'])
print("NEUTRAL:", d[b'NONE'])

print("CLASS@", datetime.now())

with open('data\\tweets\\0-58_part1\\results_part1.csv', 'w', encoding="utf-8") as csvfile:
    for i in range(len(hillary_tweets)):
        csvfile.writelines(','.join([hillary_users[i], str(hillary_dates[i]), str(hillary_stances[i])+'\n']))

print("SAVED@", datetime.now())