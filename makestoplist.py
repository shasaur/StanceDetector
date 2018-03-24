import csv
from nltk.corpus import wordnet as wn

def removeSubset(a, s):
    return [x for x in a if x not in s]

def isAdjective(word):
    synonyms = wn.synsets(word)

    adjective = False
    for s in range(len(synonyms)):
        if synonyms[s].lemma_names()[0] == word and (synonyms[s].pos() == 'a' or synonyms[s].pos() == 's'):
            adjective = True

    return adjective

words = []
accuracies = []

with open('data\\stop-words\\sentiment-occurance-popularity-index.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        accuracies.append(row[3])
        words.append(row[0])

manual_stop_words = []
for i in range(len(words)):
    if float(accuracies[i]) <= 0.5:
        manual_stop_words.append(words[i])

#manual_stop_words = removeSubset(all, subset)


stop_words = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                           'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that','the',
                           'to', 'was', 'were', 'will', 'with']

# Add manual words
stop_words.extend(manual_stop_words)

# Don't add adjectives
filtered_stop_words = []
for w in stop_words:
    if not isAdjective(w):
        filtered_stop_words.append(w)

print(filtered_stop_words)