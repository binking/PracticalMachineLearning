# kNN for Kaggle

import numpy as np
import unittest
import time, operator, random

def distance(v1, v2):
    v1 = np.array(v1); v2 = np.array(v2)
    'Compute squared euclidean distance'
    gap = v1 - v2
    return np.dot(gap.T, gap)

def featureNormalize(X):
    X = np.mat(X)
    aver = X.mean(0)
    stderror = X.std(0) + 1
    X = np.divide(X - aver, stderror) # Standard error should be a spare matrix!!! No Way
    return (X, aver, stderror)

def nnClassifier(trainingSet, trainingLabel, newSample):
    trainingSet = np.array(trainingSet)
    trainingLabel = np.array(trainingLabel)
    newSample = np.array(newSample)
    dist = np.array([distance(t, newSample) for t in trainingSet])
    nn = dist.argmin()
    return trainingLabel[nn]

def knnClassifier(trainingSet, trainingLabel, newSample, k=5):
    trainingSet = np.array(trainingSet)
    #trainingLabel = np.array(trainingLabel)
    newSample = np.array(newSample)
    dist = np.array([distance(t, newSample) for t in trainingSet]) # Calculate distances between all training sample and the only one new sample and dist.shape should be (m,)
    indices = dist.argsort() # After argsort(), indices represents indices that index nearest neibor
    classCount={}                                        
    for i in range(k):  
        voteIlabel = trainingLabel[indices[i]]  
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #print(sortedClassCount)
    return sortedClassCount[0][0]
    #print(indices)
    #numCount = {}
    #for i in range(k):
        #numCount[trainingLabel[i]] = numCount.get(trainingLabel[i], 0) + 1
    #print(numCount)
    #sortedNumCount = sorted(numCount.items(), key=operator.itemgetter(1), reverse=True)
    #print(sortedNumCount)
    #return sortedNumCount[0][0]
    #knn_labels = trainingLabel[indices].tolist() # knn_labels represents the list of k nearest neibors, but they might same
    #print(knn_labels)
    #for index in range(k):
    #    if knn_labels.count(knn_labels[index]) > int(k/2.0):
            # the neighbor emerges over 50%
    #        return knn_labels[index]
    #return knn_labels[0] # if no neibor occupied 50%, return 1NN

def loadData(filename, skip=0):
    dataset = []
    loadFromCsv = np.loadtxt(filename, skiprows = skip, delimiter = ',')
    m = loadFromCsv.shape[0]
    for row in loadFromCsv:
        dataset.append([elem for elem in row])
    return dataset

def evaluate(dataset, labels, classifier):
    print("**************      Training trainset with "+classifier.__name__+"     **************")
    dataset = np.array(dataset)
    labels = np.array(labels)
    m,n = np.shape(dataset)
    error = 0
    for i in range(m):
        predict = classifier(dataset, labels, dataset[i, :])
        real = labels[i]
        print("Real Label: ", real, "\tPredict Label: ", predict)
        if labels[i] != predict:
            error += 1
    return error/m

def saveCsv(file1, file2):
    fr = open(file1, 'r')
    fw = open(file2, 'w')
    fw.write('"ImageId","Label"\n')
    for line in fr.readlines():
        num = line.strip().split('.')[0]
        fw.write('%d,"%s"\n'%(i, num))
        i += 1
    fw.close()
    fr.close()
    return 0

def splitDataset(dataset, ratio=0.8):
    size = len(dataset)
    train_indices = list(range(size));train_set=[];test_set=[] # Using index to represents dataset
    for i in range(int((1-ratio)*size)):
        randIndex = int(random.uniform(0, len(train_indices)))
        test_set.append(dataset[randIndex])
        del(train_indices[i])
    for i in train_indices:
        train_set.append(dataset[i])
    return train_set, test_set


def main():
    #print("**************Let's get training dataset**************")
    # Processing data : 1. Split training set; 2. feature normalize
    start_time = time.time()
    dataset = loadData('train.csv', 32000)  # traindataset should contain data and labels
    print("Loading data finished")
    train_set, val_set = splitDataset(dataset, 0.8) # Split raw dataset into 8/2 parts
    print("Spliting data finished")
    train_data = [row[1:] for row in train_set]; train_labels = [row[0] for row in train_set]
    val_data = [row[1:] for row in val_set]; val_labels = [row[0] for row in val_set]
    val_size = len(val_data)
    # Features normalization for Training set
    train_data, aver, stderror = featureNormalize(train_data)
    val_data = (val_data - aver) / stderror
    print("Processing data costs :", time.time() - start_time)
    
    print("**************Let's get started recognize number**************")
    # 1NN time
    #nn_time = time.time()
    #errorRateNN = evaluate(train_data, train_labels, nnClassifier)
    #print("Error rate for NN is ", errorRateNN)
    #print("Evaluating costs of 1NN :", time.time() - nn_time)
    # validation
    #errorRateValNN = evaluate(val_data, val_labels, nnClassifier)
    #print("Validation error rate for NN is ", errorRateValNN)
    # kNN time
    knn_time = time.time()
    #errorRateKNN = evaluate(train_data, train_labels, knnClassifier)
    print("Error rate for KNN is ", errorRateKNN)
    print("Evaluating KNN costs : ", time.time() - knn_time)
    # validation
    errorRateValKNN = evaluate(val_data, val_labels, knnClassifier)
    print("Validation error rate for KNN is ", errorRateValKNN)
    #print("**************Let's get predict test dataset**************")
    #test_time = time.time()
    #testdataset = loadTestData('test.csv', 1)
    # Features normalization for Training set
    #testSet = (testSet - aver) / stderror
    #print("The size of test set is ", len(testSet))
    #print("Loading test set costs : ", time.time() - test_time)
    #m = len(testSet); result = []
    #for i in range(m):
    #    result.append(nnClassifier(trainingSet, trainLabels, testSet[i]))
    #np.savetxt('tmp.csv', testSet)
    #saveCsv('tmp.csv', 'result.csv')
'''
Processing data costs : 34.16424298286438
**************Let's get started recognize number**************
**************      Training trainset with knnClassifier     **************
Error rate for KNN is  0.05811773528308961
Evaluating KNN costs :  1808.9044625759125
**************      Training trainset with knnClassifier     **************
Validation error rate for KNN is  0.0955

'''

if __name__=='__main__':
    main()
