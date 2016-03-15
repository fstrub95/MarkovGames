__author__ = 'julien-perolat'
import numpy as np

from enum import Enum

try:
    from itertools import izip
except ImportError:  #python3.x
    izip = zip


class DatasetFormat(Enum):
    dense = 1
    binary = 2


class DataSet(object):
    def __init__(self, inputData, outputData):

        # stateAction and reward are numpy array of size (numberExample,SizeExample).
        assert inputData.shape[0] == outputData.shape[0]
        self.inputData = inputData
        self.outputData = outputData
        self.epochCompleted = 0
        self.indexEpochCompleted = 0
        self.numberExample = inputData.shape[0]

    def InputData(self):
        return self.inputData

    def OutputData(self):
        return self.outputData

    def EpochCompleted(self):
        return self.epochCompleted

    def IndexEpochCompleted(self):
        return self.indexEpochCompleted

    def NumberExample(self):
        return self.numberExample

    def NextBatch(self,sizeBatch):
        # return a minibatch of size sizeBatch
        start = self.indexEpochCompleted
        self.indexEpochCompleted = self.indexEpochCompleted + sizeBatch

        #when all the samples are used, restart and shuffle
        if self.indexEpochCompleted > self.numberExample:
            self.epochCompleted +=1

            #inplace permutation
            permute = np.arange(self.numberExample)
            np.random.shuffle(permute)

            #shuffle data
            self.inputData  = self.inputData[permute]
            self.outputData = self.outputData[permute]

            #reset indices
            start = 0
            self.indexEpochCompleted = sizeBatch
            assert sizeBatch <= self.numberExample

        end = self.indexEpochCompleted
        return self.inputData[start:end], self.outputData[start:end]



class DatasetBuilder(object):
    def __init__(self, X = None, Y = None, Ns = None, Na = None ):

        assert X is not None
        assert Ns is not None
        assert Na is not None

        self.X = X
        self.Ns = Ns
        self.Na = Na

        if Y is not None:
            self.Y = Y
        else:
            self.Y = [0 for i in range(len(X))]

        assert len(self.X) == len(self.Y)


    def generate(self,formatEnum):
        if formatEnum == DatasetFormat.binary:
            return self.generateBinary()
        elif formatEnum == DatasetFormat.dense:
            return self.generateDense()
        else:
            raise("ton format pu du ku, mais on s en fou")


    def generateBinary(self):
        sizeDataset = len(self.X)

        input  = np.zeros((sizeDataset, self.Ns + self.Na))
        output = np.zeros((sizeDataset, 1))
        for i,(s,a),q in izip(xrange(sizeDataset), self.X, self.Y):
            #input as binary representation
            input[i, s] = 1
            input[i, self.Ns + a] = 1

            #output
            output[i, 0] = q

        return DataSet(input, output)


    def generateDense(self):
        sizeDataset = len(self.X)

        input  = np.zeros((sizeDataset, 2))
        output = np.zeros((sizeDataset, 1))
        for i,(s,a),q in izip(xrange(sizeDataset), self.X, self.Y):
            #input as dense representation
            input[i, 0] = s
            input[i, 1] = a

            #output
            output[i, 0] = q

        return DataSet(input, output)
