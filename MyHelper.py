import os
# import requests
from six.moves.urllib.request import urlretrieve
import pandas as pd
import numpy as np

loc = 'Data/'
filename='mnist_train.csv'
filename2='mnist_test.csv'
def downloadData(filename,loc):
    data_point = 'https://pjreddie.com/media/files/mnist_train.csv'
    if not os.path.exists(loc+filename):
        os.chdir(loc)
        urlretrieve(data_point, filename)
        os.chdir('..')
    data_point = 'https://pjreddie.com/media/files/mnist_test.csv'
    if not os.path.exists(loc+filename2):
        os.chdir(loc)
        urlretrieve(data_point, filename2)
        os.chdir('..')

def getData(filename,loc,flat=False,test=False):
    if test:
        test_set = pd.read_csv(loc+filename2,header=None)
        #get labels in own array
        test_lb=np.array(test_set[0])

        #one hot encode the labels
        test_lb=(np.arange(10) == test_lb[:,None]).astype(np.float32)

        #drop the labels column from training dataframe
        testX=test_set.drop(0,axis=1)

        #put in correct float32 array format
        testX=np.array(testX).astype(np.float32)

        if not flat:
            testX=testX.reshape(len(testX),28,28,1)
        X,y = testX, test_lb
    else:
        train_set = pd.read_csv(loc+filename,header=None)
        #get labels in own array
        train_lb=np.array(train_set[0])

        #one hot encode the labels
        train_lb=(np.arange(10) == train_lb[:,None]).astype(np.float32)

        #drop the labels column from training dataframe
        trainX=train_set.drop(0,axis=1)

        #put in correct float32 array format
        trainX=np.array(trainX).astype(np.float32)

        if not flat:
            trainX=trainX.reshape(len(trainX),28,28,1)
    
        X,y = trainX, train_lb
    
    return X,y