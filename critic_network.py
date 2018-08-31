import tensorflow as tf
import DDPG as ddpg
from modules import *

LEN_FREQ = 20
class Critice_Network_att(ddpg.CriticNetwork):
    def __init__(self,sess, state_dim, action_dim):
        super(Critice_Network_att, self).__init__(sess,state_dim,action_dim)

    def create_q_network(self,state_dim, action_dim):
        state_input = tf.placeholder(tf.float32, [None, state_dim, LEN_FREQ])
        action_input = tf.placeholder(tf.float32, [None, action_dim, LEN_FREQ])

        feature = attention(action_input, state_input, 0.5, scope='attention')
        q_val, net = feedforward(feature,num_units=[100, 60],scope='q_network_fc')
        return state_input, action_input, q_val, net

    def create_target_q_network(self,state_dim,action_dim,net):
        state_input = tf.placeholder(tf.float32, [None, state_dim, LEN_FREQ])
        action_input = tf.placeholder(tf.float32, [None, action_dim, LEN_FREQ])













