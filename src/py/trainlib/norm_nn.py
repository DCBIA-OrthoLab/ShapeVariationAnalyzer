import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

def print_tensor_shape(tensor, string):

# input: tensor and string to describe it

    if __debug__:
        print('DEBUG ' + string, tensor.get_shape())

def convolution2d(x, filter_shape, name, strides=[1,1,1,1], activation=tf.nn.relu, padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.variable_scope(name):
# weight variable 4d tensor, first two dims are patch (kernel) size       
# third dim is number of input channels and fourth dim is output channels
        with tf.device(ps_device):

            w_conv_name = 'w_' + name
            # in_time -> stride in time
            # filter_shape=[in_time,in_channels,out_channels]
            w_conv = tf.get_variable(w_conv_name, shape=filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
            print_tensor_shape( w_conv, 'weight shape')

            b_conv_name = 'b_' + name
            b_conv = tf.get_variable(b_conv_name, shape=[filter_shape[-1]])
            print_tensor_shape( b_conv, 'bias shape')

        with tf.device(w_device):
            conv_op = tf.nn.conv2d( x, w_conv, strides=strides, padding=padding, name='conv1_op' )
            print_tensor_shape( conv_op, 'conv_op shape')

            conv_op = tf.nn.bias_add(conv_op, b_conv, name='bias_add_op')

            if(activation):
                conv_op = activation( conv_op, name='relu_op' ) 
                print_tensor_shape( conv_op, 'relu_op shape')

            return conv_op

def convolution(x, filter_shape, name, stride=1, activation=tf.nn.relu, padding="SAME", ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.variable_scope(name):
# weight variable 4d tensor, first two dims are patch (kernel) size       
# third dim is number of input channels and fourth dim is output channels
        with tf.device(ps_device):

            w_conv_name = 'w_' + name
            # in_time -> stride in time
            # filter_shape=[in_time,in_channels,out_channels]
            w_conv = tf.get_variable(w_conv_name, shape=filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))
            print_tensor_shape( w_conv, 'weight shape')

            b_conv_name = 'b_' + name
            b_conv = tf.get_variable(b_conv_name, shape=[filter_shape[-1]])
            print_tensor_shape( b_conv, 'bias shape')

        with tf.device(w_device):
            conv_op = tf.nn.conv1d( x, w_conv, stride=stride, padding=padding, name='conv1_op' )
            print_tensor_shape( conv_op, 'conv_op shape')

            conv_op = tf.nn.bias_add(conv_op, b_conv, name='bias_add_op')

            if(activation):
                conv_op = activation( conv_op, name='relu_op' ) 
                print_tensor_shape( conv_op, 'relu_op shape')

            return conv_op

def matmul(x, out_channels, name, activation=tf.nn.relu, ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.variable_scope(name):

        in_channels = x.get_shape().as_list()[-1]

    with tf.device(ps_device):
        w_matmul_name = 'w_' + name
        w_matmul = tf.get_variable(w_matmul_name, shape=[in_channels,out_channels], dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1))

        print_tensor_shape( w_matmul, 'w_matmul shape')        

        b_matmul_name = 'b_' + name
        b_matmul = tf.get_variable(name=b_matmul_name, shape=[out_channels])        

    with tf.device(w_device):

        matmul_op = tf.nn.bias_add(tf.matmul(x, w_matmul), b_matmul)

        if(activation):
            matmul_op = activation(matmul_op)

        return matmul_op

def lstm(x, out_channels, name, keep_prob=1, activation=tf.nn.relu, ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.variable_scope(name):

        xshape = x.get_shape().as_list()
        in_channels = xshape[-1]

        with tf.device(ps_device):
            # rnn_layers = [rnn.DropoutWrapper(rnn.LSTMCell(csize, activation=activation), output_keep_prob=keep_prob) for csize in out_channels]
            rnn_layers = [rnn.DropoutWrapper(rnn.Conv1DLSTMCell(xshape[1:], 3), output_keep_prob=keep_prob) for csize in out_channels]
            multi_rnn_cell = rnn.MultiRNNCell(rnn_layers)

            # initialize state cell
            # initial_state = tf.Variable(multi_rnn_cell.zero_state(shape[0], dtype=tf.float32))
            # initial_state = multi_rnn_cell.zero_state(x.get_shape()[0], dtype=tf.float32)

        with tf.device(w_device):
            outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                               inputs=x,
                                               # initial_state=initial_state,
                                               dtype=tf.float32)

        return outputs, state

def tf_repeat(tensor, multiples, expand_dim=0):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input + 1

    Returns:
    
    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, expand_dim)
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)

        print_tensor_shape(tiled_tensor, 'tiled_tensor')
    return tiled_tensor

def mask_elem(elem):
    return tf.matmul(elem[1], elem[0])

def pool_cells_to_points(cellarray, cells_to_points):

    # print_tensor_shape(cellarray, 'cellarray')
    # print_tensor_shape(cells_to_points, 'cells_to_points')
    
    # cells_to_points_shape = cells_to_points.get_shape().as_list()
    # tiled_cellarray = tf_repeat(cellarray, [cells_to_points_shape[0],1,1])

    # num_features = cellarray.get_shape().as_list()[-1]
    # tiled_cells_to_points = tf_repeat(cells_to_points, [1, 1, num_features], expand_dim=-1)

    # elems = tf.stack([tiled_cellarray, tiled_cells_to_points], axis=1)

    # print_tensor_shape(elems, 'elems')

    # result = tf.map_fn(lambda cells: tf.matmul(cells, cells_to_points), tiled_cellarray)

    return tf.matmul(cells_to_points, cellarray)

def pool_cells_to_points_op(batch_cells, cells_to_points):
    return tf.map_fn(lambda cellarray: pool_cells_to_points(cellarray, cells_to_points), batch_cells)


def inference_rnn(series, cells_to_points=None, batch_size=1, keep_prob=1, training=False, ps_device="/cpu:0", w_device="/gpu:0"):

    with tf.name_scope('rnn'):

        # series = tf.layers.batch_normalization(series, training=training)

        print_tensor_shape(series, 'series')

        shape = series.get_shape().as_list()
        conv1 = convolution2d(series, [1, 3, shape[-1], 120], "conv1_op", strides=[1,1,1,1], padding="SAME", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
        print_tensor_shape(conv1, "Out conv1")

        conv2 = convolution2d(conv1, [1, 3, 120, 240], "conv2_op", strides=[1,1,3,1], padding="VALID", activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
        print_tensor_shape(conv2, "Out conv2")

        outpool_cells_to_points = pool_cells_to_points_op(conv2[:,:,-1,:], cells_to_points)
        print_tensor_shape(outpool_cells_to_points, 'outpool_cells_to_points')

        conv3 = convolution(outpool_cells_to_points, [1, 240, 3], "conv3_op", stride=1, padding="SAME", activation=None, ps_device=ps_device, w_device=w_device)
        print_tensor_shape(conv3, "Out conv3")

        # conv3 = convolution(conv2, [3, 240, 360], "conv3_op", stride=2, activation=tf.nn.leaky_relu, ps_device=ps_device, w_device=w_device)
        # print_tensor_shape(conv3, "Out conv3")

        # conv3 = convolution(conv2, [3, 9, 3], "conv3_op", stride=2, ps_device=ps_device, w_device=w_device)
        # print_tensor_shape(conv3, "Out conv3")
        
        # series = tf.concat([series, conv1_op], 2)

        # print_tensor_shape(series, 'series')

        # create  LSTMCells        
        #outputs, state = lstm(series, [3], "lstm1", keep_prob=keep_prob, ps_device=ps_device, w_device=w_device)
        

        #conv1_op = convolution(outputs, [shape[1], 32, shape[-1]], "conv1_op", stride=shape[1], ps_device=ps_device, w_device=w_device)
        # matmul1_op = matmul(conv2[:,:,-1,:], 3, "Matmul1", activation=None, ps_device=ps_device, w_device=w_device)
        # print_tensor_shape( matmul1_op, 'Matmul1 shape')

        return conv3



def evaluation(predictions, labels, name="accuracy"):
    #return tf.metrics.accuracy(predictions=predictions, labels=labels, name=name)
    return tf.metrics.mean_absolute_error(predictions=predictions, labels=labels, name=name)
    #return tf.metrics.root_mean_squared_error(predictions=predictions, labels=labels, name=name)
    

def training(loss, learning_rate, decay_steps, decay_rate):
    # input: loss: loss tensor from loss()
    # input: learning_rate: scalar for gradient descent
    # output: train_op the operation for training

#    Creates a summarizer to track the loss over time in TensorBoard.

#    Creates an optimizer and applies the gradients to all trainable variables.

#    The Op returned by this function is what must be passed to the
#    `sess.run()` call to cause the model to train.

  # Add a scalar summary for the snapshot loss.

  # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

  # create learning_decay
    lr = tf.train.exponential_decay( learning_rate,
                                     global_step,
                                     decay_steps,
                                     decay_rate, staircase=True )

    tf.summary.scalar('2learning_rate', lr )

  # Create the gradient descent optimizer with the given learning rate.
    # optimizer = tf.train.GradientDescentOptimizer(lr)
    # optimizer = tf.train.RMSPropOptimizer(lr)
    optimizer = tf.train.AdamOptimizer(lr)

  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def loss(logits, labels):
    
    print_tensor_shape( logits, 'logits shape')
    print_tensor_shape( labels, 'labels shape')

    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy')

    # loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

    #loss = tf.losses.mean_squared_error(predictions=logits, labels=labels)
    loss = tf.losses.absolute_difference(predictions=logits, labels=labels)



    return loss