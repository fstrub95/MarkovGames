import numpy as np

from Layer import *

LAMBDA = 1e-3

def conv2d(x, W):
    return tf.nn.conv2d(x, W,  strides=[1, 1, 1, 1], padding='VALID')

def build_conv(x, in_size, out_size):
        W_conv = weight_variable([in_size,1, 1, out_size])
        b_conv = bias_variable([out_size])
        layer = [W_conv, b_conv]

        h_conv   = tf.nn.tanh(conv2d(x, W_conv) + b_conv)
        h_conv_t = tf.transpose(h_conv, perm = [0,3,2,1])

        return h_conv_t, layer

def build_conv_out(x, in_size, out_size):
        W_conv = weight_variable([in_size,1, 1, out_size])
        b_conv = bias_variable([out_size])
        layer = [W_conv, b_conv]

        h_conv   = conv2d(x, W_conv) + b_conv
        h_conv_t = tf.transpose(h_conv, perm = [0,3,2,1])

        return h_conv_t, layer

def build_QLayer(x):
    shape = x.get_shape()
    Qa, Qb = tf.split(2, 2, x)
    Qa = tf.reshape(Qa, [-1, int(shape[1])])
    Qb = tf.reshape(Qb, [-1, int(shape[1])])
    return Qa, Qb

def defQFunction(list_size, stateNextState, keep_prob):
    input_reshape  = tf.reshape(stateNextState, [-1, list_size[0], 2, 1])

    conv_out = input_reshape
    layers = []
    for i in xrange(0, len(list_size)-2):
        conv_out, layer = build_conv(conv_out, list_size[i],list_size[i+1])
        layers.append(layer)

    conv_out, layer = build_conv_out(conv_out, list_size[-2],list_size[-1])
    layers.append(layer)

    #Get the Q values from the network
    Qa, Qb = build_QLayer(conv_out)
    return Qa, Qb, layers

def regularizationl2QFunction(layers):
    reg = 0
    for layer in layers:
        reg = reg + tf.nn.l2_loss(layer[0]) + tf.nn.l2_loss(layer[1])
    return reg

def defPolicy(list_size, state, keep_prob):
    layers = [Layer(list_size[i],list_size[i+1],"") for i in range(len(list_size)-1)]
    pi = buildOutput(layers,state)
    pi = tf.nn.softmax(pi)
    return pi, layers

def regularizationl2Pi(layers):
    reg = 0
    for layer in layers:
        reg = reg + tf.nn.l2_loss(layer.w) + tf.nn.l2_loss(layer.b)
    return reg

def residualsJ0(Q0a, Q0b, pi0, pi1, turnS_, action, reward0, gamma):
    # print "Q0a"
    # print Q0a.get_shape()
    # print "Q0b"
    # print Q0b.get_shape()
    # print "pi0"
    # print pi0.get_shape()
    # print "pi1"
    # print pi1.get_shape()
    # print "turnS_"
    # print turnS_.get_shape()
    # print "action"
    # print action.get_shape()
    # print "reward0"
    # print reward0.get_shape()

    Qa = tf.reduce_sum(tf.mul(Q0a, action), reduction_indices=1, keep_dims=True)

    Qbpi0 = tf.reduce_sum(tf.mul(Q0b, pi0), reduction_indices=1, keep_dims=True)
    Qbpi1 = tf.reduce_sum(tf.mul(Q0b, pi1), reduction_indices=1, keep_dims=True)

    Qbstar = tf.reduce_max(Q0b, reduction_indices=1, keep_dims=True)

    residualJ0mean = Qa - reward0 - gamma * ( tf.mul(turnS_,Qbpi1) + tf.mul(1-turnS_,Qbpi0) )
    residualJ0opt  = Qa - reward0 - gamma * ( tf.mul(turnS_,Qbpi1) + tf.mul(1-turnS_,Qbstar) )


    return residualJ0opt, residualJ0mean

def residualsJ1(Q1a, Q1b, pi0, pi1, turnS_, action, reward1, gamma):
    # print "Q0a"
    # print Q1a.get_shape()
    # print "Q0b"
    # print Q1b.get_shape()
    # print "pi0"
    # print pi0.get_shape()
    # print "pi1"
    # print pi1.get_shape()
    # print "turnS_"
    # print turnS_.get_shape()
    # print "action"
    # print action.get_shape()
    # print "reward0"
    # print reward1.get_shape()

    Qa = tf.reduce_sum(tf.mul(Q1a, action), reduction_indices=1, keep_dims=True)

    Qbpi0 = tf.reduce_sum(tf.mul(Q1b, pi0), reduction_indices=1, keep_dims=True)
    Qbpi1 = tf.reduce_sum(tf.mul(Q1b, pi1), reduction_indices=1, keep_dims=True)

    Qbstar = tf.reduce_max(Q1b, reduction_indices=1, keep_dims=True)

    residualJ1mean = Qa - reward1 - gamma * (tf.mul(1-turnS_,Qbpi0) + tf.mul(turnS_,Qbpi1))
    residualJ1opt  = Qa - reward1 - gamma * (tf.mul(1-turnS_,Qbpi0) + tf.mul(turnS_,Qbstar))

    return residualJ1opt, residualJ1mean

def l2loss(x):
    y = tf.square(x)
    # y = tf.reduce_sum(y, reduction_indices=1, keep_dims=False)
    y = tf.reduce_mean(y) #############" est-il utile de renormaliser? Est-ce renormalise pendant la descente de gradient?

    return y

class NNQ_BR_MG(object):
    def __init__(self,list_size, datasetFormat, gamma, garnet):
        self.datasetFormat = datasetFormat
        self.Na = garnet.a
        self.Ns = garnet.s


        self.stateNextState   = tf.placeholder(tf.float32, shape=[None, list_size[0] , 2])
        self.nextState   = tf.placeholder(tf.float32, shape=[None, list_size[0], 1])

        self.action = tf.placeholder(tf.float32, shape=[None, list_size[-1]])
        self.reward0  = tf.placeholder(tf.float32, shape=[None, 1])
        self.reward1  = tf.placeholder(tf.float32, shape=[None, 1])

        self.turnS = tf.placeholder(tf.float32, shape=[None, 1])
        self.turnS_ = tf.placeholder(tf.float32, shape=[None, 1])

        self.gamma =  tf.constant(gamma)

        ######################## Variables for dropout ########################
        self.keep_prob_Q = tf.placeholder(tf.float32)
        self.keep_prob_Pi = tf.placeholder(tf.float32)


        self.Q0a, self.Q0b, layersQ0 = defQFunction(list_size,self.stateNextState, self.keep_prob_Q)
        self.Q1a, self.Q1b, layersQ1 = defQFunction(list_size,self.stateNextState, self.keep_prob_Q)

        self.pi0, layersPi0 = defPolicy(list_size, tf.reshape(self.nextState, [-1,list_size[0]]), self.keep_prob_Pi)
        self.pi1, layersPi1 = defPolicy(list_size, tf.reshape(self.nextState, [-1,list_size[0]]), self.keep_prob_Pi)

        self.residualJ0opt, self.residualJ0mean = residualsJ0(self.Q0a, self.Q0b, self.pi0, self.pi1, self.turnS_, self.action, self.reward0, self.gamma)
        self.residualJ1opt, self.residualJ1mean = residualsJ1(self.Q1a, self.Q1b, self.pi0, self.pi1, self.turnS_, self.action, self.reward1, self.gamma)

        #pick the best Q value of s_


        #compute the loss with the reward
        self.loss = l2loss(self.residualJ0opt) + l2loss(self.residualJ0mean) + l2loss(self.residualJ1opt) + l2loss(self.residualJ1mean)
        #self.loss = tf.nn.l2_loss(self.residualJ0opt) + tf.nn.l2_loss(self.residualJ0mean), tf.nn.l2_loss(self.residualJ1opt) + tf.nn.l2_loss(self.residualJ1mean))
        self.regularizer = regularizationl2QFunction(layersQ0) + regularizationl2QFunction(layersQ1) + regularizationl2Pi(layersPi0) + regularizationl2Pi(layersPi1)

        self.loss_reg = self.loss + LAMBDA*self.regularizer
        #Misc

        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss_reg)

        self.sess = tf.Session()
        self.writer = tf.train.SummaryWriter("/home/julien-perolat/Desktop/Projet NIPS/MarkovGames/out", self.sess.graph.as_graph_def(add_shapes=True))

        self.sess.run(tf.initialize_all_variables())



    def getDatasetFormat(self):
        return self.datasetFormat


    def eval(self, dataset):
        Q0 = tf.reduce_sum(tf.mul(self.Q0a, self.action), reduction_indices=1, keep_dims=True)
        Q1 = tf.reduce_sum(tf.mul(self.Q1a, self.action), reduction_indices=1, keep_dims=True)

        pi0 = tf.reduce_sum(tf.mul(self.pi0, self.action), reduction_indices=1, keep_dims=True)
        pi1 = tf.reduce_sum(tf.mul(self.pi1, self.action), reduction_indices=1, keep_dims=True)

        resQO, resQ1, respi0, respi1 = self.sess.run((Q0, Q1, pi0, pi1), feed_dict={self.stateNextState: np.concatenate((dataset.State(), dataset.NextState()), axis=2),
                                                               self.nextState : dataset.NextState(), self.action: dataset.Action(),
                                                               self.reward0 : dataset.Reward0(), self.reward1 : dataset.Reward1(),
                                                               self.turnS_: dataset.TurnNextState()})
        return resQO, resQ1, respi0, respi1


    def minimizeBellmanResidual(self, dataset, nEpoch = 100, nMiniBatch = 10, print_error = False):
        N_iter = int(nEpoch*dataset.NumberExample()/nMiniBatch)
        for i in xrange(N_iter):

            # Creating the mini-batch
            batch_state, batch_action, batch_nextState, batch_reward0, batch_reward1, batch_turnState, batch_turnNextState = dataset.NextBatch(nMiniBatch)
            batch_stateNextState = np.concatenate((batch_state, batch_nextState), axis=2)


            # running one step of the optimization method on the mini-batch
            #self.sess.run(self.optimizer, feed_dict={self.sa: batch_sa, self.s_: batch_s__, self.r: batch_r})
            self.sess.run(self.optimizer,
                          feed_dict={self.stateNextState: batch_stateNextState, self.nextState : batch_nextState,
                                     self.action: batch_action, self.reward0 : batch_reward0, self.reward1 : batch_reward1,
                                     self.turnS_: batch_turnNextState})

            if (i%(int(dataset.NumberExample()/nMiniBatch)) == 0) and print_error:
                # train error computation

                # test error computationdataset.stateAction
                errBatch = self.sess.run(self.loss_reg, feed_dict={self.stateNextState: np.concatenate((dataset.State(), dataset.NextState()), axis=2),
                                                               self.nextState : dataset.NextState(), self.action: dataset.Action(),
                                                               self.reward0 : dataset.Reward0(), self.reward1 : dataset.Reward1(),
                                                               self.turnS_: dataset.TurnNextState()})

                print "####"
                print "step %d, empirical training residual %g"%(i, errBatch/dataset.NumberExample())

        errBatch = self.sess.run(self.loss, feed_dict={self.stateNextState: np.concatenate((dataset.State(), dataset.NextState()), axis=2),
                                                        self.nextState : dataset.NextState(), self.action: dataset.Action(),
                                                        self.reward0 : dataset.Reward0(), self.reward1 : dataset.Reward1(),
                                                        self.turnS_: dataset.TurnNextState()})
        errBatch_reg = self.sess.run(self.loss_reg, feed_dict={self.stateNextState: np.concatenate((dataset.State(), dataset.NextState()), axis=2),
                                                        self.nextState : dataset.NextState(), self.action: dataset.Action(),
                                                        self.reward0 : dataset.Reward0(), self.reward1 : dataset.Reward1(),
                                                        self.turnS_: dataset.TurnNextState()})
        return errBatch, errBatch_reg