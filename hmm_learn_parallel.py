from hmmlearn.hmm import GaussianHMM
import numpy as np
from multiprocessing import Pool
from sklearn.externals import joblib

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

def readInData(num=500, testNum=100, testLabels=['0', '4', '5', '6'], all=False):
    testStartIndex = 28711

    x = open('fer2013.csv', 'r').readlines()
    pictures = [i.split(',') for i in x]

    if all:
        picObservations = [[int(j) for j in i[1].split()] for i in pictures if (i[0] in testLabels and i[2] == "Training\n")]
        labels = [i[0] for i in pictures if (i[0] in testLabels and i[2] == "Training\n")]

        testPictures = [[int(j) for j in i[1].split()] for i in pictures if (i[0] in testLabels and i[2] == "PublicTest\n")]
        groundTruth = [i[0] for i in pictures if (i[0] in testLabels and i[2] == "PublicTest\n")]
    else:
        picObservations = [[int(j) for j in i[1].split()] for i in pictures[1:num] if (i[0] in testLabels and i[2] == "Training\n")]
        labels = [i[0] for i in pictures[1:num] if (i[0] in testLabels and i[2] == "Training\n")]

        testPictures = [[int(j) for j in i[1].split()] for i in pictures[testStartIndex:testStartIndex + testNum] if (i[0] in testLabels and i[2] == "PublicTest\n")]
        groundTruth = [i[0] for i in pictures[testStartIndex:testStartIndex + testNum] if (i[0] in testLabels and i[2] == "PublicTest\n")]

    testNum = len(testPictures)

    return picObservations, labels, testPictures, groundTruth, testNum

def separateData(picObservations, testPictures):
    newPicObservations = rowPics2Mat(picObservations)
    newTestPictures = rowPics2Mat(testPictures)
    obs0 = [newPicObservations[i] for i in range(len(newPicObservations)) if labels[i] == '0']
    obs1 = [newPicObservations[i] for i in range(len(newPicObservations)) if labels[i] == '1']
    obs2 = [newPicObservations[i] for i in range(len(newPicObservations)) if labels[i] == '2']
    obs3 = [newPicObservations[i] for i in range(len(newPicObservations)) if labels[i] == '3']
    obs4 = [newPicObservations[i] for i in range(len(newPicObservations)) if labels[i] == '4']
    obs5 = [newPicObservations[i] for i in range(len(newPicObservations)) if labels[i] == '5']
    obs6 = [newPicObservations[i] for i in range(len(newPicObservations)) if labels[i] == '6']
    return obs0, obs1, obs2, obs3, obs4, obs5, obs6, newTestPictures

def myGauFit(obs):
    return GaussianHMM(n_components=48, covariance_type='full', n_iter=100).fit(obs)

def scoreModels(models, newTestPictures, testNum, testLabels, verbose=True):

    acc = 0

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
        if predicted == groundTruth[picChecked]:
            acc += 1
        if verbose:
            print "actual: ", groundTruth[picChecked]
            print '-----------------'

    total_num = len(newTestPictures)
    print 'acc: ', float(acc)/total_num

if __name__ == "__main__":
    num = 500
    testNum = 500

    testLabels = ['0', '1', '2', '3', '4', '6']

    print 'reading in data'
    picObservations, labels, testPictures, groundTruth, testNum = readInData(num, testNum, testLabels)
    print 'separating data'
    obs0, obs1, obs2, obs3, obs4, obs5, obs6, newTestPictures = separateData(picObservations, testPictures)
    observations = [obs0, obs1, obs2, obs3, obs4, obs6]
    print 'fitting gauModels'
    gauModels = list(Pool(len(observations)).map(myGauFit, observations))

    for i in testLabels:
        joblib.dump(gauModels[i], 'models/model' + i + '.pkl')

    print 'did gauModels'
    scoreModels(gauModels, newTestPictures, testNum, testLabels)