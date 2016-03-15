__author__ = 'julien-perolat'
from DataSet import *
from MDP_Garnet import *
from tools import *
from NeuralNet import *


def NNToArray(fApp, Ns, Na):
    lsa=[[(s,a) for a in xrange(Na)] for s in xrange(Ns)]
    lsa_list, size = merge_(lsa)

    datasetEval = DatasetBuilder(X=lsa_list, Ns=Ns, Na=Na).generate(fApp.getDatasetFormat())
    Qlist = fApp.eval(datasetEval)

    Q = np.zeros((Ns,Na))
    for (s,a),q in izip(lsa_list,Qlist):
        Q[s,a] = q
    return Q


def fittedQ(nIteration, fApp, batch, gamma, garnet):
    # nIteration is the number of iteration
    # fApp is an object containing the regressor
    # batch is the batch of data
    # gamma is gamma
    Q_list_array = []
    error_list = {"errBellmanResidual":[],"errDiffQstarQpi":[]}
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
        print "#########################################"
        print "Iteration %d"%(i)
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
        fApp.learn(datasetTrain, nEpoch=5)
        Q = NNToArray(fApp,garnet.s,garnet.a)
        Q_list_array.append(Q)
        errDiffQstarQpi = garnet.l2errorDiffQstarQpi(Q, gamma)
        errBellmanResidual = garnet.l2errorBellmanResidual(Q, gamma)

        error_list["errDiffQstarQpi"].append(errDiffQstarQpi)
        error_list["errBellmanResidual"].append(errBellmanResidual)

        print "######################################### Erreur exact"
        print "errDiffQstarQpi %f"%(errDiffQstarQpi)
        print "errBellmanResidual %f"%(errBellmanResidual)
    return fApp,Q_list_array
Ns = 100
Na = 10
Nb = 5
sparsity = 0.9
nIteration = 30
fApp = NNQ([Ns+Na,20,1], DatasetFormat.binary)
garnet = Garnet_MDP(Ns, Na, Nb, sparsity, Nb)
batch = garnet.uniform_batch_data(1000)
gamma = 0.99


fApp,Q_list = fittedQ(nIteration, fApp, batch, gamma, garnet)

