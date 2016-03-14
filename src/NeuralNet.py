__author__ = 'julien-perolat'

import tensorflow as tf
import numpy as np
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

class NNQ(object):
    def __init__(self,list_size, datasetFormat):
        self.datasetFormat = datasetFormat

        self.x = tf.placeholder(tf.float32, shape=[None, list_size[0]])
        self.t = tf.placeholder(tf.float32, shape=[None, list_size[-1]])

        self.layers = [Layer(list_size[i],list_size[i+1]) for i in xrange(len(list_size)-1)]

        y_out = self.x
        for layer in self.layers[:-1]:
            y_out = tf.matmul(y_out, layer.w) + layer.b
            y_out = tf.nn.relu(y_out)

        y_out = tf.matmul(y_out, self.layers[-1].w) + self.layers[-1].b
        self.y_out = y_out

        self.loss = tf.nn.l2_loss(self.t-self.y_out)
        self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def getDatasetFormat(self):
        return self.datasetFormat


    def eval(self,dataset):
        res = self.sess.run(self.y_out, feed_dict={self.x: dataset.StateAction()})
        return list(res)

    def learn(self, dataset, nEpoch = 10, nMiniBatch = 20, learningRate = 0.01):
        for i in xrange(nEpoch*dataset.NumberExample()):
            # Creating the mini-batch
            batch_xs, batch_ys = dataset.NextBatch(nMiniBatch)

            # running one step of the optimization method on the mini-batch
            self.sess.run(self.optimizer, feed_dict={self.x: batch_xs, self.t: batch_ys})

            if i%(dataset.NumberExample()) == 0:
                # train error computation
                err = self.sess.run(self.loss, feed_dict={self.x: dataset.inputData, self.t: dataset.outputData})
                err /= dataset.NumberExample()

                # test error computationdataset.stateAction
                errMiniBatch, normYOut = self.sess.run((self.loss,tf.nn.l2_loss(self.y_out)), feed_dict={self.x: batch_xs, self.t: batch_ys})
                print "####"
                print "step %d, training err %g"%(i, errMiniBatch)
                print "step %d, training err normalised %g"%(i, errMiniBatch/normYOut)