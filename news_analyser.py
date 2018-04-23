import json
from datetime import datetime
from deyfeatures import DeyFeatures

import matplotlib.pyplot as plt

def load_news_data(path):
    contents = []
    times = []

    data = json.load(open(path, 'r'))
    print("Total Entries:", len(data))

    for news in range(len(data)):
        article = data[news]

        contents.append(article["fields"]["bodyText"])

        time = datetime.strptime(article["webPublicationDate"], "%Y-%m-%dT%H:%M:%SZ")
        times.append(time)

    # for news in range(len(data)):
    #     index = 0
    #     while True:
    #         try:
    #             article = data[news]["results"][index]
    #         except:
    #             print("error at", index)
    #             break
    #
    #         contents.append(article["fields"]["bodyText"])
    #
    #         time = datetime.strptime(article["webPublicationDate"], "%Y-%m-%dT%H:%M:%SZ")
    #         times.append(time)
    #
    #         index += 1

    return contents, times

def get_sentiments_scores(content, classifier):

    #preprocess
    tokens, stemmed_tokens, unnormalised_tokens = classifier.preprocessSingle(content, True)
    print(tokens)

    #get scores
    mpqa = 0
    mpqa += classifier.calculate_mpqa(tokens[0], False, False)
    mpqa += classifier.calculate_mpqa(stemmed_tokens[0], False, False)

    swn = classifier.calculate_swn_score(tokens[0], False)

    clinton = 0
    trump = 0
    for t in tokens[0]:
        if t == 'clinton':
            clinton+=1
        elif t == 'trump':
            trump+=1

    return mpqa, swn, clinton, trump, len(tokens[0])

contents, times = load_news_data("data\\news\\theguardian-complete.json")

# Sort by date
temp = [[x,y] for x, y in zip(contents, times)]
temp.sort(key=lambda x: x[1])
contents = [x[0] for x in temp]
times = [x[1] for x in temp]

results = {"datetime":[], "sentiment_mpqa":[], "sentiment_swn":[],
           "clinton_count":[], "trump_count":[],
           "word_count":[]}

classifier = DeyFeatures()
classifier.initialise(contents)

with open("data\\news\\theguardian-scores.csv", 'w') as file:
    file.write("datetime,mpqa,swn,clinton_count,trump_count,word_count\n")

for i in range(len(contents)):
    results["datetime"].append(times[i])

    print(i, contents[i])
    mpqa, swn, clinton, trump, size = get_sentiments_scores(contents[i], classifier)

    results["sentiment_mpqa"].append(mpqa)
    results["sentiment_swn"].append(swn)
    results["clinton_count"].append(clinton)
    results["trump_count"].append(trump)
    results["word_count"].append(size)

    with open("data\\news\\theguardian-scores.csv", 'a') as file:
        file.write(','.join([str(results["datetime"][i]),
                             str(results["sentiment_mpqa"][i]),
                             str(results["sentiment_swn"][i]),
                             str(results["clinton_count"][i]),
                             str(results["trump_count"][i]),
                             str(results["word_count"][i])])+'\n')


fig, ax = plt.subplots()

#datetimes = [].append(x["datetime"] for x in results)
#scores = [].append(x["sentiment_mpqa"]/float(x["word_count"])  for x in results)

plt.plot(results["datetime"], [x/float(y) for x,y in zip(results["sentiment_mpqa"], results["word_count"])],
         c = "g")
plt.plot(results["datetime"], [x/float(y) for x,y in zip(results["sentiment_swn"], results["word_count"])],
         c = "#BB00EE")
plt.plot(results["datetime"], [x/float(y) for x,y in zip(results["clinton_count"], results["word_count"])],
         c = "b")
plt.plot(results["datetime"], [x/float(y) for x,y in zip(results["trump_count"], results["word_count"])],
         c = "r")
plt.gcf().autofmt_xdate()


plt.show()