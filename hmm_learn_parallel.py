from hmmlearn2.hmm import GaussianHMM
import numpy as np
from multiprocessing import Pool
from sklearn.externals import joblib
import csv, multiprocessing

def rowPics2Mat(pictures):
    '''
    converts a list of numbers into a 48x48 matrix
    '''
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

def strToMat(str):
    pic = [int(i) for i in str.split()]
    current = []
    i = 0
    while i < 48:
        current.append(pic[48 * i:48 * i + 48])
        i += 1
    #current = np.matrix(current)
    return current

def readInData(num=500, testNum=100, testLabels=['0', '4', '5', '6'], all_samples=False, test_all=False):
    '''
    read in data and split into training and testing sets
    return observations and labels in separate lists
    '''

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

def separateData(picObservations, testPictures, testLabels, makeEvenCounts=False):
    '''
    separate data by different emotion labels
    '''

    newPicObservations = rowPics2Mat(picObservations)
    newTestPictures = rowPics2Mat(testPictures)
    result = []
    testResult = []
    for i in range(len(newTestPictures)):
        testResult.append([np.concatenate(newTestPictures[i]), [48]])
    for label in testLabels:
        picture = np.concatenate([newPicObservations[i] for i in range(len(newPicObservations)) if labels[i] == label])
        lengths = [len(newPicObservations[i]) for i in range(len(newPicObservations)) if labels[i] == label]
        result.append([picture, lengths])
    if makeEvenCounts:
        minimum = min([len(i[1]) for i in result])
        result = [[i[0][:minimum*48], i[1][:minimum]] for i in result]
    observations = result
    #observations = [[newPicObservations[i] for i in range(len(newPicObservations)) if labels[i] == label] for label in testLabels]
    for i, j in zip(observations, testLabels):
        print len(i[1]), 'pics in label', j

    return result, testResult

def myGauFit(obs):
    '''
    fitting function to use multiprocessing, runs the Gaussian Hidden Markov Model function
    :param obs: tuple containing an array of picture matrices, their lengths, and the label for the model being fit
    :return: a trained Hidden Markov Model
    '''
    print 'starting model', obs[1]
    result = GaussianHMM(n_components=48, covariance_type='full', n_iter=100).fit(obs[0][0], lengths=obs[0][1], modelLabel=obs[1])
    print 'completed model', obs[1]
    return result

def tuplesToConfusion(tuples, testLabels, cluster=False, clusters=['']):
    '''
    converts tuples containing predictions and ground truth to a confusion matrix
    :return: a confusion matrix
    '''

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
    '''
    prints a confusion matrix using text formatting
    '''
    if cluster:
        testLabels = [i[0] for i in clusters]
    lineLen = len("   | \'" + "\' | \'".join(testLabels) + "\' |")
    print "   | \'" + "\' | \'".join(testLabels) + "\' |"
    print '-' * lineLen
    for i, j in enumerate(arr):
        print "\'" + testLabels[i] + "\'|" + '|'.join([' ' * (1 + (len(k) == 1) - (len(k) >= 4)) + k + ' ' * (2 - (len(k) == 5) - (len(k) >= 3)) for k in j]) + '|'
        print '-' * lineLen

def scoreModels(models, newTestPictures, testNum, testLabels, groundTruth, verbose=True, cluster=False, clusters=[], scoresToCSV=False):
    '''
    scores test data with a list of models and uses the most likely model as the prediction
    prints accuracy results
    :param cluster: boolean that determines if the results should be grouped by clusters
    :return: tuples containing predictions and ground truth
    '''

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

        results = [i.score(newTestPictures[picChecked][0], newTestPictures[picChecked][1]) for i in models]
        answer = {}
        for i, j in zip(results, testLabels):
            answer[i] = j
            if verbose:
                print j, i
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

    total_num = testNum
    print 'total pics scored:', total_num
    print 'acc: ', float(acc)/total_num
    if not(cluster):
        print 'top 2 acc: ', float(top2Acc) / total_num
        print 'top 3 acc: ', float(top3Acc) / total_num
    return confusionTuples

if __name__ == "__main__":
    num = 350  # Number of training samples to read in
    testNum = 100  # Number of testing samples to read in
    numThreads = 4  # Number of threads to use in cut and stitch

    all_samples = False  # Train and Test all samples
    test_all = True  # Test all samples
    cluster = False  # Note, cluster overrides top2Acc and top3Acc, group
                     # predictions into clusters
    doSave = False  # Save models to disk
    scoresToCSV = False  # Save scores in a CSV
    makeEvenCounts = False  # Make the labels have the same number of training
                            # samples

    clusters = [('2', '3', '0'), ('1'), ('4'), ('6')]
    testLabels = ['0', '2', '3', '4', '5', '6']

    if all_samples:
        num = 28711
    print 'running with', num, 'pictures'

    print 'reading in data'
    picObservations, labels, testPictures, groundTruth, testNum = readInData(num, testNum, testLabels, all_samples=all_samples, test_all=test_all)

    print 'separating data'
    observations, newTestPictures = separateData(picObservations, testPictures, testLabels, makeEvenCounts)
    observations = zip(observations, testLabels)

    print 'fitting gauModels'
    gauModelsCut = []
    gauModels = []

    import time
    start = time.time()

    timingTest = False
    if timingTest:

        cutstart = time.time()
        for obs in observations:
            gauModelsCut.append(GaussianHMM(n_components=48, covariance_type='diag', n_iter=100).parallelFit(obs[0], obs[1], numThreads))
        cutend = time.time()
        print 'cut and stitch time', cutend - cutstart

        basestart = time.time()
        gauModels = list(Pool(min(multiprocessing.cpu_count(), len(observations))).map(myGauFit, observations))
        baseend = time.time()
        print 'base time', baseend - basestart

    else:
        cutAndStitch = False
        if cutAndStitch:
            for obs in observations:
                gauModels.append(GaussianHMM(n_components=48, covariance_type='diag', n_iter=100).parallelFit(obs[0], obs[1], numThreads))
        else:
            gauModels = list(Pool(min(multiprocessing.cpu_count(), len(observations))).map(myGauFit, observations))

    end = time.time()
    print 'total time to fit', end - start

    print 'scoring gauModels'
    confusionTuplesCut = []
    if timingTest:
        print 'cut and stitch score'
        confusionTuplesCut = scoreModels(gauModelsCut, newTestPictures, testNum, testLabels, groundTruth,
                                  verbose=False, cluster=cluster, clusters=clusters, scoresToCSV=scoresToCSV)
        print 'base score'
    confusionTuples = scoreModels(gauModels, newTestPictures, testNum, testLabels, groundTruth,
                                  verbose=False, cluster=cluster, clusters=clusters, scoresToCSV=scoresToCSV)

    print 'generating confusion matrix'
    confusionMatrix = []
    if timingTest:
        print 'cut and stitch matrix'
        confusionMatrixCut = tuplesToConfusion(confusionTuplesCut, testLabels, cluster, clusters)
        printConfusion(confusionMatrixCut, testLabels, cluster, clusters)
        print 'base matrix'
    confusionMatrix = tuplesToConfusion(confusionTuples, testLabels, cluster, clusters)
    printConfusion(confusionMatrix, testLabels, cluster, clusters)

    if doSave:
        print 'saving'
        for index, i in enumerate(testLabels):
            joblib.dump(gauModels[index], 'models/model-' + i + '.pkl')
