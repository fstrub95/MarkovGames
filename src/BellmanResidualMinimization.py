__author__ = 'julien-perolat'
# from DataSetBellmanResidual import *
#from DataSet import *
from DataSetBRGeneric import *
from MDP_Garnet import *
from tools import *


Ns = 50
Na = 10
Nb = 3
noSamples = Ns * Na * 2# * Nb * 10
sparsity = 1.0
gamma = 0.9

# from NeuralNetBellmanResidual import *
#
#
# def NNToArray(fApp, Ns, Na):
#     lsa=[[(s,a) for a in xrange(Na)] for s in xrange(Ns)]
#     lsa_list, size = merge_(lsa)
#
#     datasetEval = DatasetBuilderBellmanResidual(SA=lsa_list, Ns=Ns, Na=Na).generate(fApp.getDatasetFormat())
#     Qlist = fApp.eval(datasetEval)
#
#
#     Q = np.zeros((Ns,Na))
#     for (s,a),q in izip(lsa_list,Qlist):
#         Q[s,a] = q
#     return Q
#
#
# def minimizationBellmanResidual(fApp, batch, gamma, garnet):
#     # nIteration is the number of iteration
#     # fApp is an object containing the regressor
#     # batch is the batch of data
#     # gamma is gamma
#
#     Q_list_array = []
#     slist, alist, rlist, s_list = zip(*batch)
#     salist = zip(slist, alist)
#
#     # evaluation of next state value
#     datasetsas_r = DatasetBuilderBellmanResidual(SA = salist, R = rlist, S_ = s_list, Ns=garnet.s, Na=garnet.a)\
#          .generate(fApp.getDatasetFormat())
#
#     #fApp.eval(datasetsas_r)
#
#     #
#     fApp.minimizeBellmanResidual(datasetsas_r)
#     Q = NNToArray(fApp,garnet.s,garnet.a)
#     Q_list_array.append(Q)
#     return fApp, Q_list_array
#
#
#
#
#
# def experimentconv(Ns,Na,Nb,noSamples,sparsity,gamma):
#     garnet = Garnet_MDP(Ns, Na, Nb, sparsity, Ns)
#     batch = garnet.uniform_batch_data(noSamples)
#     fApp = NNQBellmanResidual([Ns+Na,32,1], DatasetFormat.binary, gamma, garnet)
#     fApp, Q_list = minimizationBellmanResidual(fApp, batch, gamma, garnet)
#     return garnet.l2errorDiffQstarQpi(Q_list[0], gamma)
#
#
#
# l=[experimentconv(Ns,Na,Nb,noSamples,sparsity,gamma) for i in range(50)]
# print sum(l) / float(len(l))
# print l


###############################################################


# from NeuralNetBellmanResidualBis import *
# def NNToArray_(fApp, Ns, Na):
#     lsa=[[(s,a) for a in xrange(Na)] for s in xrange(Ns)]
#     lsa_list, size = merge_(lsa)
#
#     datasetEval = DatasetBuilderBellmanResidual(SA=lsa_list, Ns=Ns, Na=Na).generate(fApp.getDatasetFormat())
#     Qlist = list(fApp.eval(datasetEval.InputData())[:,0])
#
#
#     Q = np.zeros((Ns,Na))
#     for (s,a),q in izip(lsa_list,Qlist):
#         Q[s,a] = q
#     return Q
#
#
# def minimizationBellmanResidual(fApp, batch, gamma, garnet):
#     # nIteration is the number of iteration
#     # fApp is an object containing the regressor
#     # batch is the batch of data
#     # gamma is gamma
#
#     Q_list_array = []
#     slist, alist, rlist, s_list = zip(*batch)
#     salist = zip(slist, alist)
#
#     # evaluation of next state value
#     datasetsas_r = DatasetBuilderBellmanResidual(SA = salist, R = rlist, S_ = s_list, Ns=garnet.s, Na=garnet.a)\
#          .generate(fApp.getDatasetFormat())
#
#     #fApp.eval(datasetsas_r)
#
#     #
#     fApp.minimizeBellmanResidual(datasetsas_r, gamma)
#     Q = NNToArray_(fApp,garnet.s,garnet.a)
#     Q_list_array.append(Q)
#     return fApp, Q_list_array
#
# def experiment(Ns,Na,Nb,noSamples,sparsity,gamma):
#     garnet = Garnet_MDP(Ns, Na, Nb, sparsity, Ns)
#     batch = garnet.uniform_batch_data(noSamples)
#     fApp = NNQBR([Ns+Na, 32, 1], DatasetFormat.binary, gamma)
#     fApp, Q_list = minimizationBellmanResidual(fApp, batch, gamma, garnet)
#     return garnet.l2errorDiffQstarQpi(Q_list[0], gamma)
#
#
#
# l=[experiment(Ns,Na,Nb,noSamples,sparsity,gamma) for i in range(50)]
# print sum(l) / float(len(l))
# print l


###############################################################
from NeuralNetBellmanResidualBis import *
def NNToArray_(fApp, Ns, Na):
    lsa=[[(s,a) for a in xrange(Na)] for s in xrange(Ns)]
    lsa_list, size = merge_(lsa)
    slist, alist = zip(*lsa_list)

    datasetEval = DatasetBuilderBRMDPGeneric(S=slist, A=alist, Ns=Ns, Na=Na).generate(fApp.getDatasetFormat())
    Qlist = list(reshape(fApp.eval(datasetEval),(-1)))

    Q = np.zeros((Ns,Na))
    for (s,a),q in izip(lsa_list,Qlist):
        Q[s,a] = q
    return Q


def minimizationBellmanResidual(fApp, batch, gamma, garnet):
    # nIteration is the number of iteration
    # fApp is an object containing the regressor
    # batch is the batch of data
    # gamma is gamma

    Q_list_array = []
    slist, alist, rlist, s_list = zip(*batch)

    # evaluation of next state value
    datasetsas_r = DatasetBuilderBRMDPGeneric(S = slist, A = alist, R = rlist, S_ = s_list, Ns=garnet.s, Na=garnet.a)\
         .generate(fApp.getDatasetFormat())

    #fApp.eval(datasetsas_r)

    #
    fApp.minimizeBellmanResidual(datasetsas_r)
    Q = NNToArray_(fApp,garnet.s,garnet.a)
    Q_list_array.append(Q)
    return fApp, Q_list_array

from NN_BR_MDP import *

garnet = Garnet_MDP(Ns, Na, Nb, sparsity, Ns)
batch = garnet.uniform_batch_data(noSamples)
fApp = NNQ_BR_MDP([Ns, 32, Na], DatasetFormat.binary, gamma, garnet)

fApp, Q_list = minimizationBellmanResidual(fApp, batch, gamma, garnet)
print garnet.l2errorDiffQstarQpi(Q_list[0], gamma)