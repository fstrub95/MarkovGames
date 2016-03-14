__author__ = 'julien-perolat'
from DataSetBellmanResidual import *
#from DataSet import *
from MDP_Garnet import *
from tools import *
from NeuralNetBellmanResidual import *


def NNToArray(fApp, Ns, Na):
    lsa=[[(s,a) for a in xrange(Na)] for s in xrange(Ns)]
    lsa_list, size = merge_(lsa)

    datasetEval = DatasetBuilderBellmanResidual(SA=lsa_list, Ns=Ns, Na=Na).generate(fApp.getDatasetFormat())
    Qlist = fApp.eval(datasetEval)

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
    salist = zip(slist, alist)

    # evaluation of next state value
    datasetsas_r = DatasetBuilderBellmanResidual(SA = salist, R = rlist, S_ = s_list, Ns=garnet.s, Na=garnet.a).generate(fApp.getDatasetFormat())
    fApp.minimizeBellmanResidual(datasetsas_r)
    Q = NNToArray(fApp,garnet.s,garnet.a)
    Q_list_array.append(Q)
    return fApp, Q_list_array


    return fApp
Ns = 100
Na = 5
Nb = 1
sparsity = 0.5
nIteration = 30
gamma = 0.9
garnet = Garnet_MDP(Ns, Na, Nb, sparsity, Nb)

fApp = NNQBellmanResidual([Ns+Na,50,1], DatasetFormat.binary, gamma, garnet)

batch = garnet.uniform_batch_data(1000)



fApp,Q_list = minimizationBellmanResidual(fApp, batch, gamma, garnet)


print garnet.l2error(Q_list[0], gamma)

