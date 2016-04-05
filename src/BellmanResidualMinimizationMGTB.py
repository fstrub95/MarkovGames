__author__ = 'julien-perolat'
# from DataSetBellmanResidual import *
#from DataSet import *
from DataSetBRGenericMG import *
from SG_TB_GS_Garnet import *
from tools import *


Ns = 100
Na = 10
Nb = 2
noSamples = Ns * Na * 1
sparsity = 0.9
gamma = 0.9




###############################################################
from NN_BR_MG import *


def NNToArray_(fApp, Ns, Na):
    lsa=[[(s,a) for a in xrange(Na)] for s in xrange(Ns)]
    lsa_list, size = merge_(lsa)
    slist, alist = zip(*lsa_list)

    datasetEval = DatasetBuilderBRMGGeneric(S=slist, A=alist, Ns=Ns, Na=Na).generate(fApp.getDatasetFormat())
    resQ0, resQ1, respi0, respi1 = fApp.eval(datasetEval)
    resQO = list(np.reshape(resQ0, (-1)))
    resQ1 = list(np.reshape(resQ1, (-1)))
    respi0 = list(np.reshape(respi0, (-1)))
    respi1 = list(np.reshape(respi1, (-1)))

    Q0 = np.zeros((Ns,Na))
    Q1 = np.zeros((Ns,Na))
    Pi0 = np.zeros((Ns,Na))
    Pi1 = np.zeros((Ns,Na))

    for (s,a), q0, q1, pi0, pi1 in izip(lsa_list,resQO,resQ1,respi0,respi1):
        Q0[s,a] = q0
        Q1[s,a] = q1
        Pi0[s,a] = pi0
        Pi1[s,a] = pi1
    return Q0, Q1, Pi0, Pi1


def minimizationBellmanResidual(fApp, batch, gamma, garnet, N):
    # nIteration is the number of iteration
    # fApp is an object containing the regressor
    # batch is the batch of data
    # gamma is gamma

    slist, alist, turn_slist, r0list, r1list, s_list, turn_s_list = zip(*batch)

    # evaluation of next state value
    datasetsas_r = DatasetBuilderBRMGGeneric(S = slist,
                                             A = alist,
                                             R0 = r0list,
                                             R1 = r1list,
                                             S_ = s_list,
                                             turnS = turn_slist,
                                             turnS_ = turn_s_list,
                                             Ns = garnet.s,
                                             Na = garnet.a)\
        .generate(fApp.getDatasetFormat())
    #print datasetsas_r.TurnNextState()

    #fApp.eval(datasetsas_r)
    err_min = 1e20
    #
    for i in range(N):
        err, err_reg = fApp.minimizeBellmanResidual(datasetsas_r, nEpoch = 50, nMiniBatch = 5)
        if err < err_min:
            err_min = err

    Q0, Q1, Pi0, Pi1 = NNToArray_(fApp,garnet.s,garnet.a)
    return fApp, err, err_reg, Q0, Q1, Pi0, Pi1




# garnet = Garnet_SG_TB_GS(Ns, Na, Nb, sparsity, "S_linear_T2")
# batch = garnet.uniform_batch_data(noSamples)
# # print batch
# listNashError = []
# listBellmanResidual = []
# listFApp = []
# N_trial = 20
#
# for j in range(N_trial):
#     print "###################### new trial ######################"
#     print j
#     fApp = NNQ_BR_MG([Ns, Na], DatasetFormat.binary, gamma, garnet)
#     fApp, err, err_reg, Q0, Q1, Pi0, Pi1 = minimizationBellmanResidual(fApp, batch, gamma, garnet, 1)
#     print err, err_reg
#     err0, err1 = garnet.l2errorDiffQstarQpi(garnet.merge_policy(Pi0, Pi1), gamma)
#     print err0, err1
#     listNashError.append((err0, err1))
#     listBellmanResidual.append((err, err_reg))
#     listFApp.append(fApp)
#
# print sorted(zip(listBellmanResidual,listNashError))
#
# save([listNashError, listBellmanResidual], "resExperiment")


garnet = Garnet_SG_TB_GS(Ns, Na, Nb, sparsity, "S_linear_T2")
batch = garnet.uniform_batch_data(noSamples)
# print batch
listNashError = []
listBellmanResidual = []
listFApp = []

N_trial = 20

for j in range(N_trial):
    print "###################### new trial ######################"
    print j
    fApp = NNQ_BR_MG([1, int(Ns/2), int(Ns/2), Na], DatasetFormat.dense, gamma, garnet)
    fApp, err, err_reg, Q0, Q1, Pi0, Pi1 = minimizationBellmanResidual(fApp, batch, gamma, garnet, 1)
    print err, err_reg
    err0, err1 = garnet.l2errorDiffQstarQpi(garnet.merge_policy(Pi0, Pi1), gamma)
    print err0, err1
    listNashError.append((err0, err1))
    listBellmanResidual.append((err, err_reg))
    listFApp.append(fApp)

print sorted(zip(listBellmanResidual,listNashError))

save([listNashError, listBellmanResidual], "resExperiment")
