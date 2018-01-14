import tensorflow as tf
import tensorflow.contrib.slim as slim

initializer = tf.truncated_normal_initializer(stddev=0.02)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1+ leak)
        f2 = 0.5 * (1-leak)
        return f1*x+f2*abs(x)

def set_initializer(_initializer):
    global initializer
    initializer = _initializer

def conv2d(x,channels,kernel_size,name,activation=lrelu,summary=True):
    global initializer
    conv = slim.conv2d(x, channels,[kernel_size,kernel_size],padding="SAME",\
            biases_initializer=None,activation_fn=lrelu,scope=name,\
            weights_initializer=initializer)
    if summary:
        tf.summary.histogram(name,conv)
    else:
        print "NOT_SUMMARY:",name
    return conv

def downsample(x,channels,kernel_size,rate,name,activation=lrelu,summary=True):
    conv = conv2d(x,channels/4,kernel_size,name,activation=activation,summary=summary)
    downsampled = tf.space_to_depth(conv,rate)
    return downsampled

def upsample(x,channels,kernel_size,rate,name,activation=lrelu,summary=True): 
    conv = conv2d(x,channels*4,kernel_size,name,activation=activation,summary=summary)
    upsampled = tf.depth_to_space(conv,rate)
    return upsampled

def loss_mse(x,y,name):
    loss = tf.reduce_mean(tf.square(x-y))
    tf.summary.scalar(name,loss)
    return loss