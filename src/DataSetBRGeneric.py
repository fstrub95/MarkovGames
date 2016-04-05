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


class DataSetBRMDPGeneric(object):
    def __init__(self, state, action, nextState, reward, shapeMDP, typeDataset):

        # stateAction and reward are numpy array of size (numberExample,SizeExample).
        assert state.shape[0] == reward.shape[0] == action.shape[0] == nextState.shape[0]

        self.state = state
        self.action = action
        self.nextState = nextState
        self.reward = reward


        self.epochCompleted = 0
        self.indexEpochCompleted = 0

        self.numberExample = reward.shape[0]

        self.shapeMDP = shapeMDP
        self.typeDataset = typeDataset # (Ns,Na)

    def State(self):
        return self.state

    def Action(self):
        return self.action

    def NextState(self):
        return self.nextState

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
            self.state  = self.state[permute]
            self.action  = self.action[permute]
            self.nextState  = self.nextState[permute]
            self.reward = self.reward[permute]

            #reset indices
            start = 0
            self.indexEpochCompleted = sizeBatch
            assert sizeBatch <= self.numberExample

        end = self.indexEpochCompleted
        return self.state[start:end], self.action[start:end], self.nextState[start:end], self.reward[start:end]




class DatasetBuilderBRMDPGeneric(object):
    def __init__(self, S = None, A = None, R = None, S_ = None, Ns = None, Na = None ):

        assert S is not None
        assert A is not None
        assert Ns is not None
        assert Na is not None

        self.S = S
        self.A = A
        self.Ns = Ns
        self.Na = Na

        if S_ is not None:
            self.S_ = S_
        else:
            self.S_ = [0 for i in range(len(S))]

        if R is not None:
            self.R = R
        else:
            self.R = [0 for i in range(len(S))]

        assert len(self.S) == len(self.A) == len(self.R) == len(self.S_)


    def generate(self,formatEnum):
        if formatEnum == DatasetFormat.binary:
            return self.generateBinary()
        elif formatEnum == DatasetFormat.dense:
            return self.generateDense()
        else:
            print("ton format pu du ku, mais on s en fou")


    def generateBinary(self):
        sizeDataset = len(self.S)

        state  = np.zeros((sizeDataset, self.Ns , 1))
        action = np.zeros((sizeDataset, self.Na))
        nextState = np.zeros((sizeDataset, self.Ns , 1))
        reward = np.zeros((sizeDataset, 1))
        for i, s, a, r, s_ in izip(xrange(sizeDataset), self.S, self.A, self.R, self.S_):
            #Q(s,a)
            state[i, s, 0]   = 1
            action[i, a]     = 1
            nextState[i, s_] = 1
            #reward
            reward[i, 0] = r

        return DataSetBRMDPGeneric(state, action, nextState, reward, (self.Ns, self.Na), DatasetFormat.binary)


    # def generateDense(self):
    #     sizeDataset = len(self.SA)
    #
    #     stateAction = np.zeros((sizeDataset, 2))
    #     nextStates = np.zeros((sizeDataset, self.Na, 2))
    #     reward = np.zeros((sizeDataset, 1))
    #     for i,(s,a),r,s_ in izip(xrange(sizeDataset), self.SA, self.R, self.S_):
    #         # stateAction as dense representation
    #         stateAction[i, 0] = s
    #         stateAction[i, 1] = a
    #
    #         # next state
    #         for j in xrange(self.Na):
    #             nextStates[i, j, 0] = s_
    #             nextStates[i, j, 1] = j
    #
    #         # reward
    #         reward[i, 0] = r
    #
    #     return DataSetBellmanResidual(stateAction, nextStates, reward, (self.Ns, self.Na), DatasetFormat.dense)
