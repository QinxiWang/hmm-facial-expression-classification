from scipy import misc
import imageio
import os
import csv

#i. change every .png to matrix
#ii. sequence of matrix
#iii. train on that sequence


f_path = "./Cohn-expression-data/cohn-kanade-images/"
emotion_path = "./Cohn-expression-data/Emotion"

result = []

for subject in os.listdir(f_path):
    #print subject
    for sequence in  os.listdir(f_path + '/' + subject):
        last_file = ''
        for filename in os.listdir(f_path + '/' + subject + '/' + sequence):
            last_file = filename
            #print filename
            img = imageio.imread(f_path + '/' + subject + '/' + sequence+ '/' + filename)
        with open(emotion_path + '/' + subject + '/' + sequence + '/' + last_file[:-4] + '_emotion.txt') as f:
            print f.read()
            break

    break


#
# subject = 'S005'
# sequence = '001'
# image = '00000001.png'
#
#
# path = "./Cohn-expression-data/cohn-kanade-images/" + subject + "/" + sequence + '/' + image
#
# print imageio.imread(path)

