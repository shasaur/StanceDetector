import datetime

def saveFeaturesToCSV(name, features, predictions, stances):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    print("Saving...", timestamp)

    filename = "performance\\"+ name + '-' + timestamp + ".csv"
    file = open(filename, 'w')

    print(len(features), "of", len(stances))
    for feature_vector_index in range(len(features)):
        file.write(','.join(map(str, features[feature_vector_index])))
        file.write(',')
        file.write(predictions[feature_vector_index].decode("utf-8"))
        file.write(',')
        file.write(stances[feature_vector_index].decode("utf-8"))
        file.write("\n")

    file.close()
    print("Saved.")


def saveFeaturesToWeka(name, stance_nominals, features, stances):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    print("Saving...", timestamp)

    filename = "data\\weka\\" + name + '-' + timestamp + ".arff"
    file = open(filename, 'w')

    file.write('@RELATION ' + name)
    file.write('\n\n')

    for i in range(len(features[0])):
        if type(features[0][i]) is bool:
            file.write('@ATTRIBUTE f' + str(i) + ' {True, False}\n')
        else:
            file.write('@ATTRIBUTE f' + str(i) + ' NUMERIC\n')
    file.write('@ATTRIBUTE stance '+stance_nominals+'\n')
    file.write('\n')

    file.write('@DATA\n')
    for i in range(len(features)):
        file.write(','.join(map(str, features[i])))
        file.write(',')
        file.write(stances[i].decode("utf-8"))
        file.write("\n")

def loadFeaturesFromCSV(filename):
    print("Loading...")

    file = open(filename, 'r')

    feature_vector_lines = file.readlines()

    features = [[] for j in range(len(feature_vector_lines))]
    predictions = [[] for j in range(len(feature_vector_lines))]
    stances = [[] for j in range(len(feature_vector_lines))]

    for i in range(len(features)):
        feature_vector = feature_vector_lines[i].split(',')
        features[i] = feature_vector
        stances[i] = features[i].pop(len(feature_vector)-1).replace('\n', '')
        predictions[i] = features[i].pop(len(feature_vector)-1)

    file.close()
    print("Loaded.")

    return features, predictions, stances