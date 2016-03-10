__author__ = 'julien-perolat'
from DataSet import *
from MDP_Garnet import *
from tools import *
from NeuralNet import *


def fittedQ(nIteration, fApp, batch, gamma, garnet):
    # nIteration is the number of iteration
    # fApp is an object containing the regressor
    # batch is the batch of data
    # gamma is gamma

    slist, alist, rlist, s_list = zip(*batch)
    X = zip(slist, alist)
    lists_b = []
    for s_ in s_list:
        lists_b.append([(s_,b) for b in garnet.action_set_list()])    #lists_b = [[(s_,b1),(s_,b2),(s_,b3)],...,[(s_,b1),(s_,b2),(s_,b3)]]

    # evaluation of next state value
    lists_b_eval, separateur = merge_(lists_b) # lists_b_eval = [(s_,b),...(s_,b)]

    datasetEval = DatasetBuilder(X=lists_b_eval,
                             Ns=garnet.s,
                             Na=garnet.a)\
        .generate(fApp.getDatasetFormat())



    for i in xrange(nIteration):
        ###### building database ######
        # building possible next state-action list

        Qlist = fApp.eval(datasetEval)

        # retrieve max of Q for tuple (s_,b)
        Qlist = split_(Qlist, separateur)
        Qlist = [max(l) for l in Qlist]

        # output list
        Y= [r+gamma*q for r, q in izip(rlist,Qlist)]

        #generate dataset
        builder = DatasetBuilder(X=X, Y=Y,Ns=garnet.s, Na=garnet.a)
        datasetTrain = builder.generate(fApp.getDatasetFormat())

        # learning the next qfunction
        fApp.learn(datasetTrain)

    return fApp

nIteration = 100
fApp = NNQ([104,50,1], DatasetFormat.binary)
garnet = Garnet_MDP(100, 4, 5, 0.9, 100)
batch = garnet.uniform_batch_data(300)
gamma = 0.99


fittedQ(nIteration, fApp, batch, gamma, garnet)