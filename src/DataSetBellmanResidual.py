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


class DataSetBellmanResidual(object):
    def __init__(self, stateAction, nextStates, reward, shapeMDP, typeDataset):

        # stateAction and reward are numpy array of size (numberExample,SizeExample).
        assert stateAction.shape[0] == reward.shape[0] == nextStates.shape[0]

        self.stateAction = stateAction
        self.nextStates = nextStates
        self.reward = reward

        self.epochCompleted = 0
        self.indexEpochCompleted = 0

        self.numberExample = stateAction.shape[0]

        self.shapeMDP = shapeMDP
        self.typeDataset = typeDataset # (Ns,Na)

    def StateAction(self):
        return self.stateAction

    def NextStates(self):
        return self.nextStates

    def Reward(self):
        return self.reward

    def EpochCompleted(self):
        return self.epochCompleted

    def IndexEpochCompleted(self):
        return self.indexEpochCompleted

    def NumberExample(self):
        return self.numberExample

    def ShapeMDP(self):
        return self.shapeMDP

    def TypeDataset(self):
        return self.typeDataset


    def NextBatch(self,sizeBatch):
        if self.typeDataset == DatasetFormat.binary:
            return self.NextBatchBinary(sizeBatch)
        elif self.typeDataset == DatasetFormat.dense:
            return self.NextBatchDense(sizeBatch)
        else:
            print("ton format pu du ku, mais on s en fou")


    def NextBatchBinary(self,sizeBatch):
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
            self.stateAction  = self.stateAction[permute]
            self.nextStates  = self.nextStates[permute]
            self.reward = self.reward[permute]

            #reset indices
            start = 0
            self.indexEpochCompleted = sizeBatch
            assert sizeBatch <= self.numberExample

        end = self.indexEpochCompleted
        return self.stateAction[start:end], self.nextStates[start:end], self.reward[start:end]

    def NextBatchDense(self,sizeBatch):
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
            self.stateAction  = self.stateAction[permute]
            self.nextStates  = self.nextStates[permute]
            self.reward = self.reward[permute]

            #reset indices
            start = 0
            self.indexEpochCompleted = sizeBatch
            assert sizeBatch <= self.numberExample

        end = self.indexEpochCompleted
        return self.stateAction[start:end], self.nextStates[start:end], self.reward[start:end]



class DatasetBuilderBellmanResidual(object):
    def __init__(self, SA = None, R = None, S_ = None, Ns = None, Na = None ):

        assert SA is not None
        assert Ns is not None
        assert Na is not None

        self.SA = SA
        self.Ns = Ns
        self.Na = Na

        if S_ is not None:
            self.S_ = S_
        else:
            self.S_ = [0 for i in range(len(SA))]

        if R is not None:
            self.R = R
        else:
            self.R = [0 for i in range(len(SA))]

        assert len(self.SA) == len(self.R) == len(self.S_)


    def generate(self,formatEnum):
        if formatEnum == DatasetFormat.binary:
            return self.generateBinary()
        elif formatEnum == DatasetFormat.dense:
            return self.generateDense()
        else:
            print("ton format pu du ku, mais on s en fou")


    def generateBinary(self):
        sizeDataset = len(self.SA)

        stateAction  = np.zeros((sizeDataset, self.Ns + self.Na))
        nextStates  = np.zeros((sizeDataset, self.Na, self.Ns + self.Na))
        reward = np.zeros((sizeDataset, 1))
        for i,(s,a),r,s_ in izip(xrange(sizeDataset), self.SA, self.R, self.S_):
            # stateAction as binary representation
            stateAction[i, s] = 1
            stateAction[i, self.Ns + a] = 1

            # next state
            for j in xrange(self.Na):
                nextStates[i, j, s_] = 1
                nextStates[i, j, self.Ns + j] = 1

            # reward
            reward[i, 0] = r

        return DataSetBellmanResidual(stateAction, nextStates, reward, (self.Ns, self.Na), DatasetFormat.binary)


    def generateDense(self):
        sizeDataset = len(self.SA)

        stateAction = np.zeros((sizeDataset, 2))
        nextStates = np.zeros((sizeDataset, self.Na, 2))
        reward = np.zeros((sizeDataset, 1))
        for i,(s,a),r,s_ in izip(xrange(sizeDataset), self.SA, self.R, self.S_):
            # stateAction as dense representation
            stateAction[i, 0] = s
            stateAction[i, 1] = a

            # next state
            for j in xrange(self.Na):
                nextStates[i, j, 0] = s_
                nextStates[i, j, 1] = j

            # reward
            reward[i, 0] = r

        return DataSetBellmanResidual(stateAction, nextStates, reward, (self.Ns, self.Na), DatasetFormat.dense)
