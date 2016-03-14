__author__ = 'julien-perolat'

import tensorflow as tf
import numpy as np
from DataSetBellmanResidual import DataSetBellmanResidual
from DataSet import DataSet

class Layer:
    def __init__(self, inputSize, ouputSize):
        self.w = weight_variable([inputSize, ouputSize]) #warning regarding line/column
        self.b = bias_variable([ouputSize])
        self.a = None
        self.y = None




############# Usefull functions #################
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def buildOutput(layers,placeholder):
    y = placeholder
    for layer in layers[:-1]:
        y = tf.matmul(y, layer.w) + layer.b
        y = tf.nn.relu(y)

    y = tf.matmul(y, layers[-1].w) + layers[-1].b
    return y


class NNQBellmanResidual(object):
    def __init__(self,list_size, datasetFormat, gamma, garnet):
        self.datasetFormat = datasetFormat
        self.gamma = gamma
        self.Na = garnet.a

        # Building placeholder for s,sa and s_
        self.sa = tf.placeholder(tf.float32, shape=[None, list_size[0]])
        self.s_ = tf.placeholder(tf.float32, shape=[None, list_size[0]])
        self.r = tf.placeholder(tf.float32, shape=[None, list_size[-1]])

        # Building layers
        self.layers = [Layer(list_size[i],list_size[i+1]) for i in xrange(len(list_size)-1)]

        # Building output
        self.y_sa = buildOutput(self.layers, self.sa)
        self.y_s_ = buildOutput(self.layers, self.s_)

        self.loss = tf.nn.l2_loss(self.r + self.gamma * tf.reduce_max(tf.reshape(self.y_s_, [-1, self.Na, list_size[-1]]),reduction_indices=1)-self.y_sa)
        #self.optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(self.loss)
        self.optimizer = tf.train.AdagradOptimizer(0.1).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def getDatasetFormat(self):
        return self.datasetFormat


    def eval(self, dataset):
        res = self.sess.run(self.y_sa, feed_dict={self.sa: dataset.StateAction()})
        return list(res)

    def minimizeBellmanResidual(self, dataset, nEpoch = 300, nMiniBatch = 50):
        for i in xrange(nEpoch*dataset.NumberExample()):
            # Creating the mini-batch
            batch_sa, batch_s_, batch_r = dataset.NextBatch(nMiniBatch)
            shape = batch_s_.shape


            # running one step of the optimization method on the mini-batch
            #self.sess.run(self.optimizer, feed_dict={self.sa: batch_sa, self.s_: batch_s__, self.r: batch_r})
            self.sess.run(self.optimizer, feed_dict={self.sa: batch_sa, self.s_: np.reshape(batch_s_, (-1,batch_s_.shape[-1])), self.r: batch_r}) ###### peut-etre que les operations de reshape ne sont pas inversibles...

            if i%(dataset.NumberExample()) == 0:
                # train error computation

                # test error computationdataset.stateAction
                errMiniBatch, normYOut = self.sess.run((self.loss,tf.nn.l2_loss(self.y_sa)),
                                                       feed_dict={self.sa: dataset.StateAction(),
                                                                  self.s_: np.reshape(dataset.NextStates(),
                                                                                      (-1,dataset.NextStates().shape[-1])),
                                                                  self.r: dataset.Reward()})
                print "####"
                print "step %d, training err %g"%(i, errMiniBatch/dataset.NumberExample())
                print "step %d, norm training set %g"%(i, normYOut/dataset.NumberExample())
                print "step %d, training err normalised %g"%(i, errMiniBatch/normYOut)