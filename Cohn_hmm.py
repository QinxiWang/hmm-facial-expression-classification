import imageio
import os
from hmmlearn2.hmm import GaussianHMM
import numpy as np
from multiprocessing import Pool
from sklearn.externals import joblib
import math

#i. change every .png to matrix
#ii. sequence of matrix
#iii. train on that sequence

ALL_SUBJECTS = -1

f_path = "./Cohn-expression-data/cohn-kanade-images/"
emotion_path = "./Cohn-expression-data/Emotion"
landmark_path = "./Cohn-expression-data/Landmarks"

def readInData(numSubjects=10):

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
            currImgSeq = []
            currLandmarkSeq = []
            for filename in os.listdir(f_path + '/' + subject + '/' + sequence):
                last_file = filename
                if filename == '.DS_Store':
                    continue
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
            subjectArr.append({'pics' : currImgSeq, 'landmarks' : currLandmarkSeq, 'emotion' : emotion})
        result.append(subjectArr)
    print error,'missing emotion labels'
    return result

def landmarksToCoordinates(arr):
    result = []
    for i in arr:
        currSeq = []
        for j in i.split('\n'):
            if j:
                currSeq.append(j.strip().split())
        result.append(currSeq)
    return result

# x = landmarksToTuples(result[1][1]['landmarks'])

def XYToDifference(arr):
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
    result = []
    for subject in arr:
        for sequence in subject:
            result.append(sequence)
    return result

def separateLandmarkSequences(arr):
    landmarkList = []
    emotionList = []
    for subject in arr:
        currSubjLandmarks = []
        currSubjEmotions = []
        for sequence in subject:
            landmarks = sequence['landmarks']
            landmark_tuples = landmarksToCoordinates(landmarks)
            landmark_differences = XYToDifference(landmark_tuples)
            #landmark_differences = np.matmul(np.transpose(landmark_differences), landmark_differences)
            currSubjLandmarks.append(landmark_differences)
                                    # don't hate me for this, it's actually the only way I think
            currSubjEmotions.append(str(int(float(sequence['emotion'].strip()))))
        landmarkList.append(currSubjLandmarks)
        emotionList.append(currSubjEmotions)
    return unnestArrays(landmarkList), unnestArrays(emotionList)

def separateIntoCategories(landmarks, emotions, labels):
    return [np.concatenate([landmarks[i] for i in range(len(landmarks)) if emotions[i] == label]) for label in labels]
    #return [[landmarks[i] for i in range(len(landmarks)) if emotions[i] == label] for label in labels]

def fitLandmarks(trainingLandmarks, trainingEmotions, labels):
    observations = separateIntoCategories(trainingLandmarks, trainingEmotions, labels)

    print 'fitting models for', len(trainingLandmarks), 'sequences'
    iterations = 1000
    models = [GaussianHMM(n_iter=iterations).fit(obs) for obs in observations]
    return models

def scoreLandmarks(models, testLandmarks, testEmotions, labels):
    acc = 0
    top2Acc = 0
    top3Acc = 0
    for i in range(len(testEmotions)):
        scores = []
        answer = {}
        for model in models:
            scores.append(model.score(testLandmarks[i]))
        for k, j in zip(scores, labels):
            answer[k] = j
        # print 'predicted:', answer[max(scores)]
        # print 'actual:', testEmotions[i]
        if testEmotions[i] == answer[max(scores)]:
            acc += 1
        scores.sort(reverse=True)
        if testEmotions[i] in [answer[k] for k in scores[:2]]:
            top2Acc += 1
        if testEmotions[i] in [answer[k] for k in scores[:3]]:
            top3Acc += 1
    print 'acc:', float(acc)/len(testEmotions)
    print 'top 2 acc:', float(top2Acc)/len(testEmotions)
    print 'top 3 acc:', float(top3Acc)/len(testEmotions)
    return acc, top2Acc, top3Acc, len(testEmotions)

if __name__ == "__main__":

    num = ALL_SUBJECTS
    numFolds = 10
    labels = ['1', '2', '3', '4', '5', '6', '7']

    print 'running for', num, 'subjects'
    print 'reading in data'
    data = readInData(num)
    print 'formatting data'
    landmarks, emotions = separateLandmarkSequences(data)

    testNumIndexMin = 0
    testNum = len(landmarks) // numFolds
    testNumIndexMax = testNum

    totalAcc = 0
    totalTop2Acc = 0
    totalTop3Acc = 0
    totalTested = 0
    for fold in range(numFolds):
        print '=================='
        print 'fold', fold + 1

        trainingLandmarks = landmarks[:testNumIndexMin] + landmarks[testNumIndexMax:]
        trainingEmotions = emotions[:testNumIndexMin] + emotions[testNumIndexMax:]

        testLandmarks = landmarks[testNumIndexMin:testNumIndexMax]
        testEmotions = emotions[testNumIndexMin:testNumIndexMax]

        models = fitLandmarks(trainingLandmarks, trainingEmotions, labels)

        print 'scoring', len(testLandmarks), 'sequences'

        acc, top2Acc, top3Acc, tested = scoreLandmarks(models, testLandmarks, testEmotions, labels)

        totalAcc += acc
        totalTop2Acc += top2Acc
        totalTop3Acc += top3Acc
        totalTested += tested

        testNumIndexMin += testNum
        testNumIndexMax += testNum

    print 'totals after cross validation'
    print '-----------------------------'
    print 'scored', totalTested, 'sequences'
    print 'acc:', float(totalAcc)/totalTested
    print 'top 2 acc:', float(totalTop2Acc)/totalTested
    print 'top 3 acc:', float(totalTop3Acc)/totalTested




#
# subject = 'S005'
# sequence = '001'
# image = '00000001.png'
#
#
# path = "./Cohn-expression-data/cohn-kanade-images/" + subject + "/" + sequence + '/' + image
#
# print imageio.imread(path)

