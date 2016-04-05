__author__ = 'julien-perolat'

import numpy as np

from Layer import *



class NNQBR(object):
    def __init__(self,list_size, datasetFormat, gamma):
        self.datasetFormat = datasetFormat

        self.x = tf.placeholder(tf.float32, shape=[None, list_size[0]])
        self.t = tf.placeholder(tf.float32, shape=[None, list_size[-1]])

        ##############################
        self.x_ = tf.placeholder(tf.float32, shape=[None, list_size[0]])
        self.t_ = tf.placeholder(tf.float32, shape=[None, list_size[-1]])
        ##############################

        self.layers = [Layer(list_size[i],list_size[i+1], "layer_"+str(i)) for i in xrange(len(list_size)-1)]

        y_out = self.x
        for layer in self.layers[:-1]:
            y_out = tf.matmul(y_out, layer.w) + layer.b
            y_out = tf.nn.relu(y_out)

        y_out = tf.matmul(y_out, self.layers[-1].w) + self.layers[-1].b
        self.y_out = y_out

        ##############################
        y_out_ = self.x_
        for layer in self.layers[:-1]:
            y_out_ = tf.matmul(y_out_, layer.w) + layer.b
            y_out_ = tf.nn.relu(y_out_)

        y_out_ = tf.matmul(y_out_, self.layers[-1].w) + self.layers[-1].b
        self.y_out_ = y_out_
        ##############################

        self.loss = tf.nn.l2_loss(self.t-self.y_out)
        self.loss_ = tf.nn.l2_loss(self.t_-self.y_out_)

        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss + self.loss_)

        self.sess = tf.Session()

        self.writer = tf.train.SummaryWriter("/home/julien-perolat/Desktop/Projet NIPS/MarkovGames/out", self.sess.graph.as_graph_def(add_shapes=True))

        self.sess.run(tf.initialize_all_variables())

    def getDatasetFormat(self):
        return self.datasetFormat


    def eval(self,inputData):
        inputData = np.transpose(inputData, (0, 2, 1))
        shape = inputData.shape
        inputData = inputData.reshape((shape[0]*shape[1], shape[2]))

        res = self.sess.run(self.y_out, feed_dict={self.x: inputData})
        res = res.reshape((shape[0], shape[1]))

        return res

    def prepare_batch(self, batch_input, batch_target, gamma):
        shape = batch_input.shape

        eval = self.eval(batch_input)

        qsa = np.reshape(eval[:, 0], (shape[0], 1))
        qs_b = eval[:, 1:]

        max_qs_b = np.reshape(np.max(qs_b, 1), (shape[0], 1))
        argmax_qs_b = np.argmax(qs_b, 1)

        batch_s_b = np.transpose(batch_input[:,:,1:], (0,2,1))
        s_b = np.zeros((shape[0], shape[1]))
        for j in range(shape[0]):
            s_b[j, :] = batch_s_b[j, argmax_qs_b[j], :]

        # then we spleet the loss in two parts

        sa = np.reshape(batch_input[:, :, 0], (shape[0], shape[1]))
        r_qs_b = np.multiply(gamma, max_qs_b) + batch_target

        ## s_b is defined
        r_qsa = qsa - batch_target

        return sa, r_qs_b, s_b, r_qsa


    def minimizeBellmanResidual(self, dataset, gamma, nEpoch = 300, nMiniBatch = 20):

        for i in xrange(nEpoch*dataset.NumberExample()/nMiniBatch):

            # Creating the mini-batch
            batch_input, batch_target = dataset.NextBatch(nMiniBatch)

            sa, r_qs_b, s_b, r_qsa = self.prepare_batch(batch_input, batch_target, gamma)
            # input = np.concatenate((sa,s_b))
            # target = np.concatenate((r_qs_b,r_qsa))




            # running one step of the optimization method on the mini-batch
            # self.sess.run(self.optimizer, feed_dict={self.sa: batch_sa, self.s_: batch_s__, self.r: batch_r})
            self.sess.run(self.optimizer,
                            feed_dict={self.x: sa, self.t: r_qs_b, self.x_: s_b, self.t_: r_qsa})

            if i%(dataset.NumberExample()/nMiniBatch) == 0:
                # train error computation
                sa, r_qs_b, s_b, r_qsa = self.prepare_batch(dataset.InputData(), dataset.TargetData(), gamma)
                # test error computationdataset.stateAction
                errBatch = self.sess.run(self.loss + self.loss_, feed_dict={self.x: sa, self.t: r_qs_b, self.x_: s_b, self.t_: r_qsa})
                print "####"
                print "step %d, empirical training residual %g"%(i, errBatch/dataset.NumberExample())