import numpy as np

from Layer import *

def conv2d(x, W):
    return tf.nn.conv2d(x, W,  strides=[1, 1, 1, 1], padding='VALID')

def build_conv(x, in_size, out_size):
        W_conv = weight_variable([in_size,1, 1, out_size])
        b_conv = bias_variable([out_size])

        h_conv   = tf.nn.relu(conv2d(x, W_conv) + b_conv)
        h_conv_t = tf.transpose(h_conv, perm = [0,3,2,1]) #################""" Erreur!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        #h_conv_t   = tf.nn.dropout(h_conv_t,0.5)

        #shape (?, outsize, action_size, 1)
        return h_conv_t

def build_QLayer(x):
    shape = x.get_shape()
    Qa, Qb = tf.split(2, 2, x)
    Qa = tf.reshape(Qa, [-1, int(shape[1])])
    Qb = tf.reshape(Qb, [-1, int(shape[1])])
    return Qa, Qb


class NNQ_BR_MDP(object):
    def __init__(self,list_size, datasetFormat, gamma, garnet):
        self.datasetFormat = datasetFormat
        self.Na = garnet.a
        self.Ns = garnet.s

        self.stateNextState   = tf.placeholder(tf.float32, shape=[None, self.Ns , 2])
        self.action = tf.placeholder(tf.float32, shape=[None, self.Na])
        self.reward  = tf.placeholder(tf.float32, shape=[None, 1])
        self.gamma =  tf.constant(gamma)


        #fit convolution shape constraints
        input_reshape  = tf.reshape(self.stateNextState, [-1, self.Ns, 2, 1])

        #build convolution networks
        conv_out = input_reshape
        for i in xrange(0, len(list_size)-1):
            conv_out = build_conv(conv_out, list_size[i],list_size[i+1])

        #Get the Q values from the network
        Qa, Qb = build_QLayer(conv_out)

        #pick the best Q value of s_
        self.Qbmax = tf.reduce_max(Qb, reduction_indices=1, keep_dims=True)
        self.Qa = tf.reduce_sum(tf.mul(Qa, self.action), reduction_indices=1, keep_dims=True)



        #compute the Bellman residual
        self.output = self.Qa - self.gamma * self.Qbmax

        #compute the loss with the reward
        self.loss = tf.nn.l2_loss(self.output-self.reward)

        #Misc

        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess = tf.Session()
        self.writer = tf.train.SummaryWriter("/home/julien-perolat/Desktop/Projet NIPS/MarkovGames/out", self.sess.graph.as_graph_def(add_shapes=True))

        self.sess.run(tf.initialize_all_variables())



    def getDatasetFormat(self):
        return self.datasetFormat


    def eval(self, dataset):
        res = self.sess.run(self.Qa, feed_dict={self.stateNextState: np.concatenate((dataset.State(), dataset.NextState()), axis=2), self.action: dataset.Action()})
        return list(res)


    def minimizeBellmanResidual(self, dataset, nEpoch = 300, nMiniBatch = 20):
        N_iter = int(nEpoch*dataset.NumberExample()/nMiniBatch)
        for i in xrange(N_iter):

            # Creating the mini-batch
            batch_state, batch_action, batch_nextState, batch_reward = dataset.NextBatch(nMiniBatch)



            # running one step of the optimization method on the mini-batch
            #self.sess.run(self.optimizer, feed_dict={self.sa: batch_sa, self.s_: batch_s__, self.r: batch_r})
            self.sess.run(self.optimizer,
                          feed_dict={self.stateNextState: np.concatenate((batch_state, batch_nextState), axis=2), self.action: batch_action, self.reward : batch_reward})

            if i%(dataset.NumberExample()/nMiniBatch) == 0:
                # train error computation

                # test error computationdataset.stateAction
                errBatch = self.sess.run(self.loss,
                                         feed_dict={self.stateNextState: np.concatenate((dataset.State(), dataset.NextState()), axis=2), self.action: dataset.Action(), self.reward : dataset.Reward()})
                print "####"
                print "step %d, empirical training residual %g"%(i, errBatch/dataset.NumberExample())