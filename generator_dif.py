import tensorflow as tf
import config


class Generator(object):
    def __init__(self, node_emd_init):
        self.node_emd_init = node_emd_init

        with tf.variable_scope('generator'):
            self.embedding_matrix = tf.get_variable(name="embedding",
                                                    shape=self.node_emd_init.shape,
                                                    initializer=tf.constant_initializer(self.node_emd_init),
                                                    trainable=True)
        self.u_t = tf.placeholder(tf.int32, shape=[None])  
        self.U = tf.placeholder(tf.int32, shape=[None])  
        self.reward = tf.placeholder(tf.float32, shape=[None])
        self.p_v = tf.placeholder(tf.float32, shape=[None])

        self.u_t_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.u_t)
        self.U_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.U)
        self.score = self.p_v
        self.prob = tf.clip_by_value(tf.nn.sigmoid(self.score), 1e-5, 1)

        self.loss = -tf.reduce_mean(tf.log(self.prob) * self.reward) + config.lambda_gen * (
                tf.nn.l2_loss(self.u_t_embedding) + tf.nn.l2_loss(self.U_embedding))
        optimizer = tf.train.AdamOptimizer(config.lr_gen)
        self.g_updates = optimizer.minimize(self.loss)