import numpy as np
import datetime

def saveDatasetToWeka(dataset, stance_nominals, name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    print("Saving...", timestamp)

    filename = "data\\weka\\"+ name + '-' + timestamp + ".arff"
    file = open(filename, 'w')

    file.write('@RELATION ' + name)
    file.write('\n\n')
    file.write('@ATTRIBUTE id NUMERIC\n')
    file.write('@ATTRIBUTE target STRING\n')
    file.write('@ATTRIBUTE tweet STRING\n')
    file.write('@ATTRIBUTE stance '+stance_nominals+'\n')
    file.write('\n')
    file.write('@DATA')
    file.write('\n')


    for i in range(len(dataset)):
        # if not dataset['stance'][i].decode('UTF-8') == 'NONE':
        file.write(str(dataset['id'][i]))
        file.write(',')
        file.write("'" + dataset['target'][i].decode('UTF-8') + "'")
        file.write(',')
        file.write("'" + dataset['tweet'][i].decode('UTF-8').replace("'",' ') + "'")
        file.write(',')
        file.write(dataset['stance'][i].decode('UTF-8'))
        # if dataset['stance'][i].decode('UTF-8') == 'NONE':
        #     file.write('NONE')
        # else:
        #     file.write('OTHER')
        file.write("\n")

    file.close()
    print("Saved.")


def saveFeaturesToCSV(name, features, predictions, stances):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    print("Saving...", timestamp)

    filename = "data\\weka\\" + name + '-' + timestamp + ".arff"
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

test_all=np.genfromtxt('C:\\Users\\shasa\\scikit_learn_data\\se16-test-gold.txt', delimiter='\t', comments="((((",
                dtype={'names': ('id', 'target', 'tweet', 'stance'),
                       'formats': ('int', 'S50', 'S200', 'S10')})
hillary_train_set = test_all[test_all['target'] == b'Hillary Clinton']

saveDatasetToWeka(hillary_train_set, 'raw_test_set')
