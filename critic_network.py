import tensorflow as tf
from transformer import multihead_attention, feedforward
from hyperparams import Hyperparams as hp

def build_critic(s, a, is_training):
    """
    :param s: Tensor of shape (N * dims * C)
    :param a: Tensor of shape (N * dima * C)
    :return: tensors of train_op and inference result
    """



def update_critic(s,s_,a,r):
    target = build_critic(s,a,is_training=False)
    y = r + target
    loss = tf.y - build_critic()
