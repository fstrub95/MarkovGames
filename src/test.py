__author__ = 'julien-perolat'
import tensorflow as tf
import numpy as np
N_exp = 20
Na = 10
size_vect = 100
mat = np.zeros((N_exp, Na, size_vect))
for n in xrange(N_exp):
    for a in xrange(Na):
        for s in xrange(size_vect):
            mat[n,a,s] = a + Na*n


print mat
mat = mat.reshape((-1,size_vect))

x = tf.placeholder(tf.float32, shape=[None, size_vect])
b = tf.ones([size_vect,1])
tf.Variable(b)

y = tf.matmul(x, b)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
res = sess.run(tf.reduce_max(tf.reshape(y, [-1, Na]), reduction_indices=1), feed_dict={x: mat})
print res