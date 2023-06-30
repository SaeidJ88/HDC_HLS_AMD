

#ref=https://github.com/Raisony/HD_ISOLET


import copy
import pickle
import random
import argparse
import numpy as np
import csv
# Loading training and testing data
def dataLoader(dataset):
    with open('./'+ dataset + '.pkl', 'rb') as f:
         Xtr, Xts, ytr, yts = pickle.load(f)
    return Xtr, Xts, ytr, yts

# Binary search
def numToKey(value, levelList):
    if (value == levelList[-1]):
        return len(levelList)-2
    upperIndex = len(levelList) - 1
    lowerIndex = 0
    keyIndex = 0
    while (upperIndex > lowerIndex):
        keyIndex = int((upperIndex + lowerIndex)/2)
        if (levelList[keyIndex] <= value and levelList[keyIndex+1] > value):
            return keyIndex
        if (levelList[keyIndex] > value):
            upperIndex = keyIndex
            #keyIndex = int((upperIndex + lowerIndex)/2)
        else:
            lowerIndex = keyIndex
            #keyIndex = int((upperIndex + lowerIndex)/2)
    return keyIndex

def genLabel(dataset):
    label = dataset.reshape(-1).tolist()
    return label

def quantz(hdvector):
    HV = copy.deepcopy(hdvector)
    HV = np.array(HV)
    #threshold = np.mean(HV)
    threshold = 0.5
    HV[HV > threshold] = 1
    HV[HV == threshold] = 0
    HV[HV < threshold] = -1
    return HV.tolist()

def getlevelList(totalLevel, minimum, maximum):
    levelList = []
    length = maximum - minimum
    gap = length / totalLevel
    for level in range(totalLevel):
        levelList.append(minimum + level*gap)
    levelList.append(maximum)
    return levelList

def HV_encoding(HDC, baseVector, levelVector, trainingData, testingData):
    HV_train, HV_test = [], []
    levelVector = copy.deepcopy(levelVector)
    dimension = HDC.dim
    for i in range(trainingData.shape[0]):
        trainData = trainingData[i, :]
        hdv = HDC.encoding(dimension, trainData, levelVector, baseVector)
        hdv = quantz(hdv)
        #hdv[hdv >= 0 ] = 1
        #hdv[hdv == 0] = 0
        #hdv[hdv < 0] = -1
        
        HV_train.append(hdv)
    for i in range(testingData.shape[0]):
        testData = testingData[i, :]
        hdv = HDC.encoding(dimension, testData, levelVector, baseVector)
        hdv = quantz(hdv)
        #hdv[hdv >= 0 ] = 1
        #hdv[hdv == 0] = 0
        #hdv[hdv < 0] = -1
        HV_test.append(hdv)
    return np.array(HV_train), np.array(HV_test)

def checkVector(classHVs, inputHV):
    guess = list(classHVs.keys())[0]
    maximum = np.NINF
    count, checklist = {} ,[]
    for key in classHVs.keys():
        count[key] = associateSearch(classHVs[key], inputHV)
        # inner_product(classHVs[key], inputHV)
        checklist.append([key, associateSearch(classHVs[key], inputHV)])
        if (count[key] > maximum):
            guess = key
            maximum = count[key]
    checklist = sorted(checklist, key = lambda x: x[1], reverse = -1)
    return guess, checklist

def associateSearch(HV1, HV2):
    disim = np.dot(HV1, HV2)/(np.linalg.norm(HV1) * np.linalg.norm(HV2) + 0.0)
    #DISprint('disim', disim )
    return disim


#HDC model
class HyperDimensionalComputing(object):
    def __init__(self, dimension, totalPos, totalLevel, datatype, buffer, *string, cuda = False):
        self.P = totalPos
        self.Q = totalLevel
        self.dim = dimension
        self.buffer = buffer
        self.datatype = datatype
        self.levelList = getlevelList(totalLevel, self.buffer[0], self.buffer[1])
        
    def genBaseVector(self, totalPos, baseVal, dimension):
        D = dimension
        baseHVs = dict()
        indexVector = range(D)
        change = int(D/2)
        for level in range(totalPos):
            name = level
            base = np.full(D, baseVal)
            toOne = np.random.permutation(indexVector)[:change]  
            for index in toOne:
                base[index] = 1
            baseHVs[name] = copy.deepcopy(base)     
        return baseHVs

    def genLevelVector(self, totalLevel, baseVal, dimension):
        D = dimension
        levelHVs = dict()
        indexVector = range(D)
        nextLevel = int((D/2/totalLevel))
        change = int(D/2)
        for level in range(totalLevel):
            name = level
            if(level == 0):
                base = np.full(D, baseVal)
                toOne = np.random.permutation(indexVector)[:change]
            else:
                toOne = np.random.permutation(indexVector)[:nextLevel]
            for index in toOne:
                base[index] = base[index] * -1
            levelHVs[name] = copy.deepcopy(base)
        return levelHVs

    def encoding(self, dimension, label, levelHVs, baseHVs):
        HDVector = np.zeros(dimension, dtype = self.datatype)
        for keyVal in range(len(label)):
            key = numToKey(label[keyVal], self.levelList)
            baseHV = baseHVs[keyVal]
            levelHV = levelHVs[key]
            HDVector = HDVector + (baseHV * levelHV)
        return HDVector

    def genClassHV(self, classHV, inputLabels, inputHVs):
        #This creates a dict with no duplicates
        classHVs = copy.deepcopy(classHV)
        for i in range(len(inputLabels)):
            name = inputLabels[i]
            classHVs[name] = np.array(classHVs[name]) + np.array(inputHVs[i])
        return classHVs

    def oneShotTraining(self, classHVs, trainHVs, trainLabels, testHVs, testLabels,):
        retClassHVs = copy.deepcopy(classHVs)
        currAcc = self.test(retClassHVs, testHVs, testLabels)
        for index in range(len(trainLabels)):
            predict, dis_checklist = checkVector(retClassHVs, trainHVs[index])
            if not (trainLabels[index] == predict):
                retClassHVs[predict] = retClassHVs[predict] - trainHVs[index]
                retClassHVs[trainLabels[index]] = retClassHVs[trainLabels[index]] + trainHVs[index]
        return retClassHVs, currAcc

    def retraining (self, classHVs, trainHVs, trainLabels, testHVs, testLabels, n_iteration):
        bestWeight, bestacc = dict(), 0
        currClassHV = copy.deepcopy(classHVs)
        for i in range(n_iteration):
            currClassHV, _ = self.oneShotTraining(currClassHV, trainHVs, trainLabels, testHVs, testLabels,)
            currAcc = self.test(currClassHV, testHVs, testLabels)
            print('Epoch', i, 'accuracy:', currAcc)
            if currAcc > bestacc:
                bestAcc = currAcc
                bestWeight = currClassHV
        return currClassHV, currAcc, bestWeight, bestAcc

    def test(self, classHVs, testHVs, testLabels):
        correct = 0
        for index in range(len(testHVs)):
            predict, checklist = checkVector(classHVs, testHVs[index])
            if (testLabels[index] == predict):
                correct += 1
        accuracy = (correct / len(testLabels)) * 100
        return accuracy
    
    
if __name__ == '__main__':
    
    
   # parser = argparse.ArgumentParser()
   # parser.add_argument('--app_', metavar='N', type=str)
   # parser.add_argument('--iter_', metavar='N', type=int)
   # parser.add_argument('--dimension_', metavar='N', type=int)

    app_ = 'isolet'
    iter_ = 20
    dimension_ = 1024
    #args = parser.parse_args()
    
    Xtr, Xts, ytr, yts = dataLoader(app_)
    #main(dimension_, iter_, [Xtr, ytr], [Xts, yts])
    
    dimension=dimension_
    iteration =iter_;
    training = [Xtr, ytr];
    testing= [Xts, yts];
    
    HDC = HyperDimensionalComputing(dimension, totalPos = training[0].shape[1], totalLevel = 100, datatype = np.int16, buffer = [-1.0, 1.0], cuda = False)
    trainingData, testingData, trainLabel, testLabel = training[0], testing[0], genLabel(training[1]), genLabel(testing[1])
    classHV = dict([(x, np.array([0 for _ in range(dimension)])) for x in range(1, len(np.unique(testLabel)) + 1)])
    
    
    baseVector = HDC.genBaseVector(HDC.P, -1, HDC.dim)
    levelVector = HDC.genLevelVector(HDC.Q, -1, HDC.dim)
    HVector, HVector_test = HV_encoding(HDC, baseVector, levelVector, trainingData, testingData)
    
    #HVector, HVector_test = simple_HV_encoding(HDC, trainingData, testingData)
    
    classHVs = HDC.genClassHV(classHV, trainLabel , HVector)
    currWeight, currAcc = HDC.oneShotTraining(classHVs, HVector, trainLabel, HVector_test, testLabel,)
    print('One shot accuracy:', currAcc)
    print('-------- Retrain', iteration, 'epochs --------')
    currWeight, currAcc, bestWeight, bestAcc = HDC.retraining(currWeight, HVector, trainLabel , HVector_test, testLabel, iteration)






    # Define the CSV file path
    csv_file = 'baseVector.csv'
    
    # Get the maximum number of elements in any key's value
    max_elements = max(len(values) for values in baseVector.values())
    
    # Create a list of column names
    columns = list(baseVector.keys())
    
    # Flatten the dictionary structure and fill missing values with None
    flattened_data = []
    for i in range(max_elements):
        row = [baseVector[key][i] if i < len(baseVector[key]) else None for key in columns]
        flattened_data.append(row)
    
    # Write the flattened data to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)  # Write header
        writer.writerows(flattened_data)
    
    print(f"Data exported to {csv_file} successfully.")
    
    
    
    
    
    
    # Define the CSV file path
    csv_file = 'currWeight.csv'
    
    # Get the maximum number of elements in any key's value
    max_elements = max(len(values) for values in currWeight.values())
    
    # Create a list of column names
    columns = list(currWeight.keys())
    
    # Flatten the dictionary structure and fill missing values with None
    flattened_data = []
    for i in range(max_elements):
        row = [currWeight[key][i] if i < len(currWeight[key]) else None for key in columns]
        flattened_data.append(row)
    
    # Write the flattened data to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)  # Write header
        writer.writerows(flattened_data)
    
    print(f"Data exported to {csv_file} successfully.")
    
    
    
    
    # Define the CSV file path
    csv_file = 'levelVector.csv'
    
    # Get the maximum number of elements in any key's value
    max_elements = max(len(values) for values in levelVector.values())
    
    # Create a list of column names
    columns = list(levelVector.keys())
    
    # Flatten the dictionary structure and fill missing values with None
    flattened_data = []
    for i in range(max_elements):
        row = [levelVector[key][i] if i < len(levelVector[key]) else None for key in columns]
        flattened_data.append(row)
    
    # Write the flattened data to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)  # Write header
        writer.writerows(flattened_data)
    
    print(f"Data exported to {csv_file} successfully.")
    
    
    
    
    

    
    
    
    