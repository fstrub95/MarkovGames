import tensorflow as tf

class Layer:
    def __init__(self, inputSize, ouputSize, name):
            self.w = weight_variable([inputSize, ouputSize]) #warning regarding line/column
            self.b = bias_variable([ouputSize])
            self.a = None
            self.y = None


############# Usefull functions #################
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1, name ="weight")
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape, name = 'bias')
  return tf.Variable(initial)

def buildOutput(layers,placeholder):
    y = placeholder
    for layer in layers[:-1]:
        y = tf.matmul(y, layer.w) + layer.b
        y = tf.nn.tanh(y)


    y = tf.matmul(y, layers[-1].w) + layers[-1].b
    return y