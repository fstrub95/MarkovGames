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


class DataSetBRMGGeneric(object):
    def __init__(self, state, action, nextState, reward0, reward1, turnState, turnNextState, shapeMDP, typeDataset):

        # stateAction and reward are numpy array of size (numberExample,SizeExample).
        assert state.shape[0] == reward0.shape[0] == reward1.shape[0] == action.shape[0] == nextState.shape[0]

        self.state = state
        self.action = action
        self.nextState = nextState
        self.reward0 = reward0
        self.reward1 = reward1
        self.turnState = turnState
        self.turnNextState = turnNextState


        self.epochCompleted = 0
        self.indexEpochCompleted = 0

        self.numberExample = state.shape[0]

        self.shapeMDP = shapeMDP
        self.typeDataset = typeDataset # (Ns,Na)

    def State(self):
        return self.state

    def Action(self):
        return self.action

    def NextState(self):
        return self.nextState

    def Reward0(self):
        return self.reward0

    def Reward1(self):
        return self.reward1

    def TurnState(self):
        return self.turnState

    def TurnNextState(self):
        return self.turnNextState

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
            self.reward0 = self.reward0[permute]
            self.reward1 = self.reward1[permute]
            self.turnState = self.turnState[permute]
            self.turnNextState = self.turnNextState[permute]

            #reset indices
            start = 0
            self.indexEpochCompleted = sizeBatch
            assert sizeBatch <= self.numberExample

        end = self.indexEpochCompleted
        return self.state[start:end], self.action[start:end], self.nextState[start:end], self.reward0[start:end], self.reward1[start:end], self.turnState[start:end], self.turnNextState[start:end]




class DatasetBuilderBRMGGeneric(object):
    def __init__(self, S = None, A = None, R0 = None, R1 = None, S_ = None, turnS = None, turnS_ = None, Ns = None, Na = None):

        assert S is not None
        assert A is not None
        assert Ns is not None
        assert Na is not None

        self.S = S
        self.A = A
        self.Ns = Ns
        self.Na = Na

        if turnS_ is not None:
            self.turnS_ = turnS_
        else:
            self.turnS_ = [0 for i in range(len(S))]

        if turnS is not None:
            self.turnS = turnS
        else:
            self.turnS = [0 for i in range(len(S))]

        if S_ is not None:
            self.S_ = S_
        else:
            self.S_ = [0 for i in range(len(S))]

        if R0 is not None:
            self.R0 = R0
        else:
            self.R0 = [0 for i in range(len(S))]

        if R1 is not None:
            self.R1 = R1
        else:
            self.R1 = [0 for i in range(len(S))]

        assert len(self.S) == len(self.A) == len(self.R0) == len(self.S_)


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
        reward0 = np.zeros((sizeDataset, 1))
        reward1 = np.zeros((sizeDataset, 1))
        turnState = np.zeros((sizeDataset, 1))
        turnNextState = np.zeros((sizeDataset, 1))
        for i, s, a, turn_s, r0, r1, s_, turn_s_ in izip(xrange(sizeDataset), self.S, self.A, self.turnS, self.R0, self.R1, self.S_, self.turnS_):
            #Q(s,a)
            state[i, s, 0]      = 1
            action[i, a]        = 1
            nextState[i, s_, 0] = 1
            #reward
            reward0[i, 0] = r0
            reward1[i, 0] = r1
            turnState[i, 0] = turn_s
            turnNextState[i, 0] = turn_s_

        return DataSetBRMGGeneric(state, action, nextState, reward0, reward1, turnState, turnNextState, (self.Ns, self.Na), DatasetFormat.binary)


    def generateDense(self):
        sizeDataset = len(self.S)

        state  = np.zeros((sizeDataset, 1, 1))
        action = np.zeros((sizeDataset, self.Na))
        nextState = np.zeros((sizeDataset, 1, 1))
        reward0 = np.zeros((sizeDataset, 1))
        reward1 = np.zeros((sizeDataset, 1))
        turnState = np.zeros((sizeDataset, 1))
        turnNextState = np.zeros((sizeDataset, 1))
        for i, s, a, turn_s, r0, r1, s_, turn_s_ in izip(xrange(sizeDataset), self.S, self.A, self.turnS, self.R0, self.R1, self.S_, self.turnS_):
            #Q(s,a)
            state[i, 0, 0]     = s
            action[i, a]       = 1
            nextState[i, 0, 0] = s_
            #reward
            reward0[i, 0] = r0
            reward1[i, 0] = r1
            turnState[i, 0] = turn_s
            turnNextState[i, 0] = turn_s_

        return DataSetBRMGGeneric(state, action, nextState, reward0, reward1, turnState, turnNextState, (self.Ns, self.Na), DatasetFormat.binary)
