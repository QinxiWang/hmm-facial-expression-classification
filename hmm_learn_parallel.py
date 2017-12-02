from hmmlearn.hmm import GaussianHMM
import numpy as np
from multiprocessing import Pool
from sklearn.externals import joblib
import csv, multiprocessing

def rowPics2Mat(pictures):
    result = []
    for pic in pictures:
        current = []
        i = 0
        while i < 48:
            current.append(pic[48 * i:48 * i + 48])
            i += 1
        current = np.matrix(current)
        result.append(current)
    return result


# model = GaussianHMM(n_components=48, covariance_type='diag', n_iter=100).fit(newPicObservations)
# print model.score(newPicObservations[1])
# num 10: -16727.2707333
# num 50: -16242.8598496
# num 100: -15691.0567749

def readInData(num=500, testNum=100, testLabels=['0', '4', '5', '6'], all_samples=False, test_all=False):
    testStartIndex = 28711

    x = open('fer2013.csv', 'r').readlines()
    pictures = [i.split(',') for i in x]

    if all_samples:
        picObservations = [[int(j) for j in i[1].split()] for i in pictures if (i[0] in testLabels and i[2] == "Training\n")]
        labels = [i[0] for i in pictures if (i[0] in testLabels and i[2] == "Training\n")]

        testPictures = [[int(j) for j in i[1].split()] for i in pictures if (i[0] in testLabels and i[2] == "PublicTest\n")]
        groundTruth = [i[0] for i in pictures if (i[0] in testLabels and i[2] == "PublicTest\n")]
    else:
        picObservations = [[int(j) for j in i[1].split()] for i in pictures[1:num] if (i[0] in testLabels and i[2] == "Training\n")]
        labels = [i[0] for i in pictures[1:num] if (i[0] in testLabels and i[2] == "Training\n")]
        if test_all:
            testPictures = [[int(j) for j in i[1].split()] for i in pictures if
                            (i[0] in testLabels and i[2] == "PublicTest\n")]
            groundTruth = [i[0] for i in pictures if
                           (i[0] in testLabels and i[2] == "PublicTest\n")]
        else:
            testPictures = [[int(j) for j in i[1].split()] for i in pictures[testStartIndex:testStartIndex + testNum] if (i[0] in testLabels and i[2] == "PublicTest\n")]
            groundTruth = [i[0] for i in pictures[testStartIndex:testStartIndex + testNum] if (i[0] in testLabels and i[2] == "PublicTest\n")]

    testNum = len(testPictures)

    return picObservations, labels, testPictures, groundTruth, testNum

def separateData(picObservations, testPictures, testLabels):
    newPicObservations = rowPics2Mat(picObservations)
    newTestPictures = rowPics2Mat(testPictures)
    observations = [[newPicObservations[i] for i in range(len(newPicObservations)) if labels[i] == label] for label in testLabels]
    for i, j in zip(observations, testLabels):
        print len(i), 'pics in label', j

    return observations, newTestPictures

def myGauFit(obs):
    print 'starting model', obs[1]
    result = GaussianHMM(n_components=48, covariance_type='full', n_iter=100).fit(obs[0], obs[1])
    print 'completed model', obs[1]
    return result

def tuplesToConfusion(tuples, testLabels, cluster=False, clusters=['']):

    indices = {}
    size = len(testLabels)
    if cluster:
        size = len(clusters)
        for i, group in enumerate(clusters):
            for j in group:
                indices[j] = i
    else:
        for i, j in enumerate(testLabels):
            indices[j] = i
    result = [[0 for j in range(size)] for i in range(size)]
    for i, j in tuples:
        result[indices[i]][indices[j]] += 1
    
    result = [[str(result[i][j]) for j in range(size)] for i in range(size)]
    return result

def printConfusion(arr, testLabels, cluster=False, clusters=[]):
    if cluster:
        testLabels = [i[0] for i in clusters]
    lineLen = len("   | \'" + "\' | \'".join(testLabels) + "\' |")
    print "   | \'" + "\' | \'".join(testLabels) + "\' |"
    print '-' * lineLen
    for i, j in enumerate(arr):
        # this may be the worst line of code I've ever written. I hate text formatting
        print "\'" + testLabels[i] + "\'|" + '|'.join([' ' * (1 + (len(k) == 1) - (len(k) >= 4)) + k + ' ' * (2 - (len(k) == 5) - (len(k) >= 3)) for k in j]) + '|'
        print '-' * lineLen

def scoreModels(models, newTestPictures, testNum, testLabels, groundTruth, verbose=True, cluster=False, clusters=[], scoresToCSV=False):

    confusionTuples = []
    acc = 0
    top2Acc = 0
    top3Acc = 0

    count = {}
    for i in groundTruth:
        count[i] = count.setdefault(i, 0) + 1
    for j in list(count.keys()):
        print count[j], 'test pics in', j
    if scoresToCSV:
        f = open('scores.csv', 'wb')
        writer = csv.writer(f)
        writer.writerow(testLabels + ['actual'])
    for picChecked in range(testNum):
        if verbose:
            print "checking num", picChecked
            print '-----------------'

        results = [i.score(newTestPictures[picChecked]) for i in models]
        answer = {}
        for i, j in zip(results, testLabels):
            answer[i] = j
        predicted = answer[max(results)]
        if verbose:
            print "predicted: ", predicted
        actual = groundTruth[picChecked]
        if scoresToCSV:
            writer.writerow(results + [actual])
        if cluster:
            for c in clusters:
                if predicted in c and actual in c:
                    acc += 1
        else:
            results.sort(reverse=True)
            top3 = [answer[i] for i in results[:3]]
            top2 = [answer[i] for i in results[:2]]
            if actual in top3:
                top3Acc += 1
            if actual in top2:
                top2Acc += 1
            if predicted == actual:
                acc += 1
        if verbose:
            print "actual: ", actual
            print '-----------------'
        confusionTuples.append((predicted, actual))
    if scoresToCSV:
        f.close()

    total_num = len(newTestPictures)
    print 'total pics scored:', total_num
    print 'acc: ', float(acc)/total_num
    if not(cluster):
        print 'top 2 acc: ', float(top2Acc) / total_num
        print 'top 3 acc: ', float(top3Acc) / total_num
    return confusionTuples

if __name__ == "__main__":
    num = 350
    testNum = 100

    all_samples = False
    test_all = True
    cluster = False  # Note, cluster overrides top2Acc and top3Acc
    doSave = False
    scoresToCSV = True

    clusters = [('0', '1', '2', '4'), ('3'), ('6')]
    testLabels = ['0', '1', '2', '3', '4', '6']

    if all_samples:
        num = 28711
    print 'running with', num, 'pictures'

    print 'reading in data'
    picObservations, labels, testPictures, groundTruth, testNum = readInData(num, testNum, testLabels, all_samples=all_samples, test_all=test_all)

    print 'separating data'
    observations, newTestPictures = separateData(picObservations, testPictures, testLabels)
    observations = zip(observations, testLabels)

    print 'fitting gauModels'
    gauModels = list(Pool(len(observations)).map(myGauFit, observations))

    print 'scoring gauModels'
    confusionTuples = scoreModels(gauModels, newTestPictures, testNum, testLabels, groundTruth,
                                  verbose=False, cluster=cluster, clusters=clusters, scoresToCSV=scoresToCSV)

    print 'generating confusion matrix'
    confusionMatrix = tuplesToConfusion(confusionTuples, testLabels, cluster, clusters)
    printConfusion(confusionMatrix, testLabels, cluster, clusters)

    if doSave:
        print 'saving'
        for index, i in enumerate(testLabels):
            joblib.dump(gauModels[index], 'models/model-' + i + '.pkl')
