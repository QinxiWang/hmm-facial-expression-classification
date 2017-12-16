# Hidden Markov Model Facial Expression Classification

This is a Python program that is designed to use Hidden Markov Models to classify facial expressions based on images and sequences of extracted facial feature coordinates. 

## Installation 

After cloning the repo you will need to download Python 2.7 for your computer. 32 bit Python for Windows and 64 bit Python for Mac/Linux. If you would like to use a different version of Python 2.7, you will need to install the hmmlearn package for Python and replace the _hmmc.* file in the hmmlearn2 folder of our program. 

You will also need to download the datasets into the top level of our program. You can find the Kaggle dataset [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) (it will need to be given a csv extension) and the Cohn-Kanade dataset [here](http://www.pitt.edu/~emotion/ck-spread.html) (which will require you to request the database from them). 

## Classification 

After doing that, you can run the hmm_learn_parallel.py file to do emotion classification on the Kaggle dataset and Cohn_hmm.py to do emotion classification on the Cohn-Kanade dataset and see the results! 

Just a heads up, the Kaggle dataset is pretty large and even though it's parallelized, it takes a long time to run when training on more than 1000 pictures since each one is over 2000 pixels large. 

## Acknowledgements

This project was made by [Qinxi Wang](https://github.com/QinxiWang) and [Matthew Davids](https://github.com/mattdavids). 

Special thanks to [Alicia Johnson](https://github.com/ajohns24) for her help and support through Bayesian Statistics!
