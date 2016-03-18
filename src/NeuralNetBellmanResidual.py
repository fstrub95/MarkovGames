__author__ = 'julien-perolat'

import numpy as np

from Layer import *


def conv2d(x, W):
    return tf.nn.conv2d(x, W,  strides=[1, 1, 1, 1], padding='VALID')

def build_conv(x, in_size, out_size):
        W_conv = weight_variable([in_size,1, 1, out_size])
        b_conv = bias_variable([out_size])

        h_conv   = tf.nn.relu(conv2d(x, W_conv) + b_conv)
        h_conv_t = tf.transpose(h_conv, perm = [0,3,2,1])

        #h_conv_t   = tf.nn.dropout(h_conv_t,0.5)

        #shape (?, outsize, action_size, 1)
        return h_conv_t

def build_QLayer(x):

    dim = int(x.get_shape()[2]) #action size

    conv_out   = tf.reshape(x, [-1, dim])

    out_conv_t = tf.transpose(conv_out)

    partitions = np.zeros(dim, dtype=int)
    partitions[0] = 1

    slices = tf.dynamic_partition(out_conv_t, partitions.tolist(), 2)

    Qa = tf.transpose(slices[1])
    Qb = tf.transpose(slices[0])

    return Qa, Qb


class NNQBellmanResidual(object):
    def __init__(self,list_size, datasetFormat, gamma, garnet):
        self.datasetFormat = datasetFormat
        self.Na = garnet.a
        self.Ns = garnet.s

        self.input   = tf.placeholder(tf.float32, shape=[None, self.Na + self.Ns , self.Na + 1])
        self.target  = tf.placeholder(tf.float32, shape=[None, 1])
        self.gamma =  tf.constant(gamma)


        #fit convolution shape constraints
        input_reshape  = tf.reshape(self.input, [-1, self.Na + self.Ns, self.Na + 1, 1])

        #build convolution networks
        conv_out = input_reshape
        for i in xrange(0, len(list_size)-1):
            conv_out = build_conv(conv_out, list_size[i],list_size[i+1])


        #Get the Q values from the network
        Qa, Qb = build_QLayer(conv_out)

        #pick the best Q value of s_
        Qbmax = tf.reduce_max(Qb, reduction_indices=1, keep_dims=True)

        #compute the Bellman residual
        self.output = Qa - self.gamma * Qbmax

        #compute the loss with the reward
        self.loss = tf.nn.l2_loss(self.output-self.target)

        #Misc

        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess = tf.Session()
        self.writer = tf.train.SummaryWriter("/home/fstrub/Projects/MarkovGames/out", self.sess.graph.as_graph_def(add_shapes=True))

        self.sess.run(tf.initialize_all_variables())



    def getDatasetFormat(self):
        return self.datasetFormat


    def eval(self, dataset):
        res = self.sess.run(self.output, feed_dict={self.input: dataset.InputData()})
        return list(res)


    def minimizeBellmanResidual(self, dataset, nEpoch = 10, nMiniBatch = 50):

        for i in xrange(nEpoch*dataset.NumberExample()):

            # Creating the mini-batch
            batch_input, batch_target = dataset.NextBatch(nMiniBatch)



            # running one step of the optimization method on the mini-batch
            #self.sess.run(self.optimizer, feed_dict={self.sa: batch_sa, self.s_: batch_s__, self.r: batch_r})
            self.sess.run(self.optimizer,
                          feed_dict={self.input: batch_input, self.target: batch_target})

            if i%(dataset.NumberExample()) == 0:
                # train error computation

                # test error computationdataset.stateAction
                errBatch = self.sess.run(self.loss, feed_dict={self.input: dataset.InputData(), self.target: dataset.TargetData()})
                print "####"
                print "step %d, empirical training residual %g"%(i, errBatch/dataset.NumberExample())
