from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import GMMHMM
from hmmlearn.hmm import MultinomialHMM
import numpy as np

num = 300
#testNum = 100

testStartIndex = 28711
# testStartIndex = 1
testNum = 5

testLabels = ['0', '1', '2', '3', '4', '5', '6']

print 'reading in data'

x = open('fer2013.csv', 'r').readlines()
pictures = [i.split(',') for i in x]
# picObservations = [tuple([j for j in i[1].split()]) for i in pictures[1:100] if i[0] in ['0', '3', '4', '6']]

picObservations = [[int(j) for j in i[1].split()] for i in pictures[1:num] if (i[0] in testLabels and i[2] == "Training\n")]
labels = [i[0] for i in pictures[1:num] if (i[0] in testLabels and i[2] == "Training\n")]

testPictures = [[int(j) for j in i[1].split()] for i in pictures[testStartIndex:testStartIndex + testNum] if (i[0] in testLabels and i[2] == "PublicTest\n")]
groundTruth = [i[0] for i in pictures[testStartIndex:testStartIndex + testNum] if (i[0] in testLabels and i[2] == "PublicTest\n")]

testNum = len(testPictures)

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

newPicObservations = rowPics2Mat(picObservations)
newTestPictures = rowPics2Mat(testPictures)

print 'generating models'

obs0 = [newPicObservations[i] for i in range(len(newPicObservations)) if labels[i] == '0']
obs1 = [newPicObservations[i] for i in range(len(newPicObservations)) if labels[i] == '1']
obs2 = [newPicObservations[i] for i in range(len(newPicObservations)) if labels[i] == '2']
obs3 = [newPicObservations[i] for i in range(len(newPicObservations)) if labels[i] == '3']
obs4 = [newPicObservations[i] for i in range(len(newPicObservations)) if labels[i] == '4']
obs5 = [newPicObservations[i] for i in range(len(newPicObservations)) if labels[i] == '5']
obs6 = [newPicObservations[i] for i in range(len(newPicObservations)) if labels[i] == '6']

#newPicObservations.shape = (48, 48)

model0Gau = GaussianHMM(n_components=48, covariance_type='full', n_iter=100).fit(obs0)
model1Gau = GaussianHMM(n_components=48, covariance_type='full', n_iter=100).fit(obs1)
model2Gau = GaussianHMM(n_components=48, covariance_type='full', n_iter=100).fit(obs2)
model3Gau = GaussianHMM(n_components=48, covariance_type='full', n_iter=100).fit(obs3)
model4Gau = GaussianHMM(n_components=48, covariance_type='full', n_iter=100).fit(obs4)
model5Gau = GaussianHMM(n_components=48, covariance_type='full', n_iter=100).fit(obs5)
model6Gau = GaussianHMM(n_components=48, covariance_type='full', n_iter=100).fit(obs6)

try:
    model0GauMix = GMMHMM(n_components=48, covariance_type='full', n_iter=100).fit(obs0)
    model1GauMix = GMMHMM(n_components=48, covariance_type='full', n_iter=100).fit(obs1)
    model2GauMix = GMMHMM(n_components=48, covariance_type='full', n_iter=100).fit(obs2)
    model3GauMix = GMMHMM(n_components=48, covariance_type='full', n_iter=100).fit(obs3)
    model4GauMix = GMMHMM(n_components=48, covariance_type='full', n_iter=100).fit(obs4)
    model5GauMix = GMMHMM(n_components=48, covariance_type='full', n_iter=100).fit(obs5)
    model6GauMix = GMMHMM(n_components=48, covariance_type='full', n_iter=100).fit(obs6)
except ValueError:
    print 'err gaumix'

try:
    model0Multi = MultinomialHMM(n_components=48, n_iter=100).fit(obs0)
    model1Multi = MultinomialHMM(n_components=48, n_iter=100).fit(obs1)
    model2Multi = MultinomialHMM(n_components=48, n_iter=100).fit(obs2)
    model3Multi = MultinomialHMM(n_components=48, n_iter=100).fit(obs3)
    model4Multi = MultinomialHMM(n_components=48, n_iter=100).fit(obs4)
    model5Multi = MultinomialHMM(n_components=48, n_iter=100).fit(obs5)
    model6Multi = MultinomialHMM(n_components=48, n_iter=100).fit(obs6)
except ValueError:
    print 'err gaumuitl'


print 'scoring models'

Gau_acc, GauMix_acc, Multi_acc = 0,0,0

for picChecked in range(testNum):
    print "checking num", picChecked
    print '-----------------'

    try:
        resultGau0 = model0Gau.score(newTestPictures[picChecked])
        resultGau1 = model1Gau.score(newTestPictures[picChecked])
        resultGau2 = model2Gau.score(newTestPictures[picChecked])
        resultGau3 = model3Gau.score(newTestPictures[picChecked])
        resultGau4 = model4Gau.score(newTestPictures[picChecked])
        resultGau5 = model5Gau.score(newTestPictures[picChecked])
        resultGau6 = model6Gau.score(newTestPictures[picChecked])
    except ValueError:
        print 'err gau'

    try:
        resultMix0 = model0GauMix.score(newTestPictures[picChecked])
        resultMix1 = model1GauMix.score(newTestPictures[picChecked])
        resultMix2 = model2GauMix.score(newTestPictures[picChecked])
        resultMix3 = model3GauMix.score(newTestPictures[picChecked])
        resultMix4 = model4GauMix.score(newTestPictures[picChecked])
        resultMix5 = model5GauMix.score(newTestPictures[picChecked])
        resultMix6 = model6GauMix.score(newTestPictures[picChecked])
    except ValueError:
        print 'err gaumix'

    try:
        resultMulti0 = model0Multi.score(newTestPictures[picChecked])
        resultMulti1 = model1Multi.score(newTestPictures[picChecked])
        resultMulti2 = model2Multi.score(newTestPictures[picChecked])
        resultMulti3 = model3Multi.score(newTestPictures[picChecked])
        resultMulti4 = model4Multi.score(newTestPictures[picChecked])
        resultMulti5 = model5Multi.score(newTestPictures[picChecked])
        resultMulti6 = model6Multi.score(newTestPictures[picChecked])
    except ValueError:
        print 'err muti'

    answer = {}

    try:
        answer[resultGau0] = '0'
        answer[resultGau1] = '1'
        answer[resultGau2] = '2'
        answer[resultGau3] = '3'
        answer[resultGau4] = '4'
        answer[resultGau5] = '5'
        answer[resultGau6] = '6'
    except KeyError:
        print 'err gau key'

    try:
        answer[resultMix0] = '0'
        answer[resultMix1] = '1'
        answer[resultMix2] = '2'
        answer[resultMix3] = '3'
        answer[resultMix4] = '4'
        answer[resultMix5] = '5'
        answer[resultMix6] = '6'
    except KeyError:
        print 'err gauMix key'

    try:
        answer[resultMulti0] = '0'
        answer[resultMulti1] = '1'
        answer[resultMulti2] = '2'
        answer[resultMulti3] = '3'
        answer[resultMulti4] = '4'
        answer[resultMulti5] = '5'
        answer[resultMulti6] = '6'
    except KeyError:
        print 'err muitl key'

    try:
        Gau_predicted = answer[max(resultGau0, resultGau1, resultGau2, resultGau3, resultGau4, resultGau5, resultGau6)]
        print "Gau_predicted:", Gau_predicted
        if Gau_predicted == groundTruth[picChecked]:
            Gau_acc += 1
    except KeyError:
        print 'err gau key'

    try:
        GauMix_predicted = answer[max(resultMix0, resultMix1, resultMix2, resultMix3, resultMix4, resultMix5, resultMix6)]
        print "GauMix_predicted:", GauMix_predicted
        if GauMix_predicted == groundTruth[picChecked]:
            GauMix_acc += 1
    except KeyError:
        print 'err gau key'

    try:
        Multi_predicted = answer[max(resultMulti0, resultMulti1, resultMulti2, resultMulti3, resultMulti4, resultMulti5, resultMulti6)]
        print "Multi_predicted:", Multi_predicted
        if Multi_predicted  == groundTruth[picChecked]:
            Multi_acc += 1
    except KeyError:
        print 'err gau key'

    print '-----------------'

    print "actual: ", groundTruth[picChecked]
    print '-----------------'

total_num = len(newTestPictures)
print 'Gau acc: ', float(Gau_acc)/total_num
print 'GauMix acc: ', float(GauMix_acc)/total_num
print 'Multi acc: ', float(Multi_acc)/total_num

# model = GaussianHMM(n_components=48, covariance_type='diag', n_iter=100).fit(newPicObservations)
# print model.score(newPicObservations[1])
# num 10: -16727.2707333
# num 50: -16242.8598496
# num 100: -15691.0567749