import tensorflow as tf

def _hinf_op(a,b):
    """

    :param a: Nx1xd
    :param b: NxCxd
    :return: NxCx1
    """
    inv = 1./b
    return tf.reduce_max(a*inv, axis=-1)


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
    return outputs


def attention(queries, keys,
              dropout_rate,
                scope):
    qs = tf.split(queries,tf.shape(queries)[1],axis=1)
    with tf.variable_scope(scope):
        hinfx = []
        for q in qs:# for K times
            hinfx.append(tf.expand_dims(_hinf_op(q, keys),1))
        w = tf.concat(hinfx,axis=1)# NxKxC
        act = tf.matmul(w,keys) / (tf.shape(keys).as_list[-1] ** 0.5) #NxKxd
        output = act + queries
    return normalize(output)

def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    channels = tf.shape(inputs).aslist()[1]
    with tf.variable_scope(scope, reuse=reuse):

        w1 = tf.Variable(tf.truncated_normal(shape=[10, channels,num_units[0]]))
        b1 = tf.Variable()
        # Inner layer
        layer1 = tf.nn.conv1d(inputs, filters=w1, stride=1, padding='SAME', data_format='NCHW') + b1

        w2 = tf.Variable(tf.truncated_normal(shape=[10, channels, num_units[1]]))
        b2 = tf.Variable()

        outputs = tf.nn.conv1d(layer1, filters=w2, stride=1, padding='SAME', data_format='NCHW') + b2

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)
        return outputs, [w1, w2, b1, b2]

