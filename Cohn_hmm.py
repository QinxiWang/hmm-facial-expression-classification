import os
from hmmlearn2.hmm import GaussianHMM
import numpy as np
import math

ALL_SUBJECTS = -1

# the file paths for the Cohn-expression-data
f_path = "./Cohn-expression-data/cohn-kanade-images/"
emotion_path = "./Cohn-expression-data/Emotion"
landmark_path = "./Cohn-expression-data/Landmarks"

def readInData(numSubjects=10):
    '''
    reads in the emotion labels and landmarks from the paths defined above
    :param numSubjects:
    :return:
    '''
    result = []
    iter = 0
    error = 0
    for subject in os.listdir(f_path):
        if iter == numSubjects:
            break
        iter += 1
        if subject == '.DS_Store':
            continue
        subjectArr = []
        for sequence in os.listdir(f_path + '/' + subject):
            if sequence == '.DS_Store':
                continue
            last_file = ''
            last_image = []
            currImgSeq = []
            currLandmarkSeq = []
            for filename in os.listdir(f_path + '/' + subject + '/' + sequence):
                last_file = filename
                if filename == '.DS_Store':
                    continue
                #last_image = imageio.imread(f_path + '/' + subject + '/' + sequence+ '/' + filename)
                # img = imageio.imread(f_path + '/' + subject + '/' + sequence+ '/' + filename)
                # currImgSeq.append(img)

            for filename in os.listdir(landmark_path + '/' + subject + '/' + sequence):
                if filename == '.DS_Store':
                    continue
                with open(landmark_path + '/' + subject + '/' + sequence + '/' + filename) as landmark:
                    currLandmarkSeq.append(landmark.read())
            try:
                with open(emotion_path + '/' + subject + '/' + sequence + '/' + last_file[:-4] + '_emotion.txt') as f:
                    emotion = f.read()
            except IOError:
                error += 1
                continue
            subjectArr.append({'pics' : last_image, 'landmarks' : currLandmarkSeq, 'emotion' : emotion})
        result.append(subjectArr)
    print error,'missing emotion labels'
    return result

def landmarksToCoordinates(arr):
    '''
    converts lists of landmark sequences into coordinate pair lists
    :param arr: lists of landmark sequences
    :return: lists of landmark sequences in coordinate pair lists
    '''
    result = []
    for i in arr:
        currSeq = []
        for j in i.split('\n'):
            if j:
                currSeq.append(j.strip().split())
        result.append(currSeq)
    return result

def XYToDifference(arr):
    '''
    converts lists of coordinate pair lists into the same lists with each entry being the Euclidiean distance from the
    previous list's entry
    :param arr: lists of landmark sequences in coordinate pair lists
    :return: lists of landmark sequences in Euclidean distances
    '''
    prevVals = arr[0]
    result = []
    for i, j in enumerate(arr):
        currSeq = []
        for curr, prev in zip(j, prevVals):
            currSeq.append(math.sqrt((float(curr[0]) - float(prev[0])) ** 2 + (float(curr[1]) - float(prev[1])) ** 2))
        result.append(currSeq)
        prevVals = arr[i]
    return result

def unnestArrays(arr):
    '''
    converts nested lists into a single list
    :param arr: nested list
    :return: unnested list
    '''
    result = []
    for subject in arr:
        for sequence in subject:
            result.append(sequence)
    return result

def separateLandmarkSequences(arr):
    '''
    takes in lists of landmark sequences, converts them to Euclidean distance. Takes in lists of emotion strings,
    converts them to a single list
    :param arr: list of dictionaries with landmarks and emotions as their key/values
    :return: list of all landmark sequences, list of all emotion labels
    '''
    landmarkList = []
    emotionList = []
    for subject in arr:
        currSubjLandmarks = []
        currSubjEmotions = []
        for sequence in subject:
            landmarks = sequence['landmarks']
            landmark_tuples = landmarksToCoordinates(landmarks)
            landmark_differences = XYToDifference(landmark_tuples)
            currSubjLandmarks.append(landmark_differences)
            currSubjEmotions.append(str(int(float(sequence['emotion'].strip()))))
        landmarkList.append(currSubjLandmarks)
        emotionList.append(currSubjEmotions)
    return unnestArrays(landmarkList), unnestArrays(emotionList)

def separateIntoCategories(landmarks, emotions, labels, makeEvenCounts):
    '''
    separates landmark sequences into emotion categories
    :param makeEvenCounts: boolean to determine if it should throw out observations to make all categories have the same
    number of observations
    :return: list of landmark sequences
    '''
    result = []
    for label in labels:
        lands = np.concatenate([landmarks[i] for i in range(len(landmarks)) if emotions[i] == label])
        lengths = [len(landmarks[i]) for i in range(len(landmarks)) if emotions[i] == label]
        result.append([lands, lengths])
    if makeEvenCounts:
        minimum = min([len(i[1]) for i in result])
        newResult = []
        for i in result:
            val = sum(i[1][:minimum])
            newResult.append([i[0][:val], i[1][:minimum]])
        result = newResult
    return result

def fitLandmarks(trainingLandmarks, trainingEmotions, labels, makeEvenCounts=False):
    '''
    trains models on all the observations and returns the models
    :return: GaussianHMM trained models
    '''
    observations = separateIntoCategories(trainingLandmarks, trainingEmotions, labels, makeEvenCounts)

    print 'fitting models for', sum([len(obs[1]) for obs in observations]), 'sequences'
    iterations = 1000
    models = [GaussianHMM(n_iter=iterations).fit(obs[0], obs[1]) for obs in observations]
    return models

def scoreLandmarks(models, testLandmarks, testEmotions, labels, cluster=False, clusters=[], totalAcc=0, totalTop2Acc=0, totalTop3Acc=0, totalTested=0):
    '''
    scores a list of models on a list of testLandmarks and prints out the accuracy of the model
    :param cluster: boolean whether or not to cluster the results
    :param clusters: list of tuples of labels to cluster the results into
    :return: accuracy values and a list of tuples with the predicted and actual value for a confusion matrix
    '''
    acc = 0
    top2Acc = 0
    top3Acc = 0
    avgProb = 0
    confusionTuples = []
    for i in range(len(testEmotions)):
        scores = []
        answer = {}
        for model in models:
            scores.append(model.score(testLandmarks[i]))
        for k, j in zip(scores, labels):
            answer[k] = j
        predicted = answer[max(scores)]
        actual = testEmotions[i]
        confusionTuples.append((predicted, actual))

        nonNegative = [(p - min(scores))/(max(scores) - min(scores)) for p in scores]
        probs = [p/sum(nonNegative) for p in nonNegative]
        prob = max(probs)

        if cluster:
            for c in clusters:
                if predicted in c and actual in c:
                    acc += 1
        else:
            if actual == predicted:
                acc += 1
            scores.sort(reverse=True)
            if actual in [answer[k] for k in scores[:2]]:
                top2Acc += 1
            if actual in [answer[k] for k in scores[:3]]:
                top3Acc += 1
    print 'running acc:', float(totalAcc + acc)/(totalTested + len(testEmotions))
    if not(cluster):
        print 'running top 2 acc:', float(totalTop2Acc + top2Acc)/(totalTested + len(testEmotions))
        print 'running top 3 acc:', float(totalTop3Acc + top3Acc)/(totalTested + len(testEmotions))
    print 'acc:', float(acc)/len(testEmotions)
    if not(cluster):
        print 'top 2 acc:', float(top2Acc)/len(testEmotions)
        print 'top 3 acc:', float(top3Acc)/len(testEmotions)
    return acc, top2Acc, top3Acc, len(testEmotions), confusionTuples

def tuplesToConfusion(tuples, testLabels, cluster=False, clusters=['']):
    '''
    converts a list of tuples into a confusion matrix
    :param tuples: list of tuples with the predicted and actual value
    :param cluster: boolean whether or not to cluster the results
    :param clusters: list of tuples of labels to cluster the results into
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
    displays a confusion matrix based on a confusion matrix array
    :param arr: a confusion matrix
    :param cluster: boolean whether or not to cluster the results
    :param clusters: list of tuples of labels to cluster the results into
    '''
    if cluster:
        testLabels = [i[0] for i in clusters]
    lineLen = len("   | \'" + "\' | \'".join(testLabels) + "\' |")
    print "   | \'" + "\' | \'".join(testLabels) + "\' |"
    print '-' * lineLen
    for i, j in enumerate(arr):
        print "\'" + testLabels[i] + "\'|" + '|'.join([' ' * (1 + (len(k) == 1) - (len(k) >= 4)) + k + ' ' * (2 - (len(k) == 5) - (len(k) >= 3)) for k in j]) + '|'
        print '-' * lineLen

if __name__ == "__main__":

    num = ALL_SUBJECTS  # number of subjects to read in data on
    numFolds = 10  # number of folds of cross validation
    labels = ['1', '2', '3', '4', '5', '6', '7']
    clusters = [('1', '3', '6'), ('2'), ('4'), ('5'), ('7')]
    cluster = False  # group prediction results into clusters
    makeEvenCounts = False  # make the labels have the same number of training samples

    print 'running for', num, 'subjects'
    if cluster:
        print 'with clusters', clusters
    print 'reading in data'
    data = readInData(num)
    print 'formatting data'
    landmarks, emotions = separateLandmarkSequences(data)

    for i, j in zip(separateIntoCategories(landmarks, emotions, labels, makeEvenCounts), labels):
        print len(i[1]), 'sequences in label', j

    testNumIndexMin = 0
    testNum = len(landmarks) // numFolds
    testNumIndexMax = testNum

    totalAcc = 0
    totalTop2Acc = 0
    totalTop3Acc = 0
    totalTested = 0
    totalConfusion = []

    for fold in range(numFolds):
        print '=================='
        print 'fold', fold + 1

        trainingLandmarks = landmarks[:testNumIndexMin] + landmarks[testNumIndexMax:]
        trainingEmotions = emotions[:testNumIndexMin] + emotions[testNumIndexMax:]

        testLandmarks = landmarks[testNumIndexMin:testNumIndexMax]
        testEmotions = emotions[testNumIndexMin:testNumIndexMax]

        models = fitLandmarks(trainingLandmarks, trainingEmotions, labels, makeEvenCounts)

        print 'scoring', len(testLandmarks), 'sequences'

        acc, top2Acc, top3Acc, tested, confusionTuples = scoreLandmarks(models, testLandmarks, testEmotions, labels, cluster, clusters, totalAcc, totalTop2Acc, totalTop3Acc, totalTested)

        totalAcc += acc
        totalTop2Acc += top2Acc
        totalTop3Acc += top3Acc
        totalTested += tested
        totalConfusion += confusionTuples

        testNumIndexMin += testNum
        testNumIndexMax += testNum

    print 'totals after cross validation'
    print '-----------------------------'
    print 'scored', totalTested, 'sequences'
    print 'acc:', float(totalAcc)/totalTested
    if not(cluster):
        print 'top 2 acc:', float(totalTop2Acc)/totalTested
        print 'top 3 acc:', float(totalTop3Acc)/totalTested
    print 'generating confusion matrix'
    confusionMatrix = tuplesToConfusion(totalConfusion, labels, cluster, clusters)
    printConfusion(confusionMatrix, labels, cluster, clusters)

