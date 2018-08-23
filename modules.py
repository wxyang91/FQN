import tensorflow as tf

def _hinf_op(a,b):
    """

    :param a:
    :param b:
    :return:
    """
    inv = 1./b
    return tf.reduce_max(a*inv, axis=-1)

def attention(queries, keys,
              dropout_rate,
              is_training=,
                scope):
    qs = tf.split(queries,tf.shape(queries)[1],axis=1)
    with tf.variable_scope(scope):
        hinfx = []
        for q in qs:
            hinfx.append(tf.expand_dims(_hinf_op(q, keys),1))
            tf.concat(hinfx,axis=1)

if __name__ == '__main__':
    a = tf.constant([1,2,3,4,5],dtype=tf.float32)
    b = 1./a
    c = tf.constant([[1,3,5,7,8],[2,4,6,8,10]],dtype=tf.float32)
    d = tf.split(b,5)
    sess = tf.Session()
    print(sess.run([c*b,d]))