import tensorflow as tf
import config


class Discriminator(object):
    def __init__(self, node_emd_init):
        self.node_emd_init = node_emd_init

        with tf.variable_scope('discriminator'):
            self.embedding_matrix = tf.get_variable(name="embedding",
                                                    shape=self.node_emd_init.shape,
                                                    initializer=tf.constant_initializer(self.node_emd_init),
                                                    trainable=True)

        self.u_t = tf.placeholder(tf.int32, shape=[None])
        self.U = tf.placeholder(tf.int32, shape=[None])
        self.label = tf.placeholder(tf.float32, shape=[None])
        self.score = tf.placeholder(tf.float32,shape=[None])

        self.u_t_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.u_t)
        self.U_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.U)

        self.loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.score)) + config.lambda_dis * (
                tf.nn.l2_loss(self.u_t_embedding) +
                tf.nn.l2_loss(self.U_embedding))
        optimizer = tf.train.AdamOptimizer(config.lr_dis)
        self.d_updates = optimizer.minimize(self.loss)
        self.score = tf.clip_by_value(self.score, clip_value_min=-10, clip_value_max=10)
        self.reward = tf.log(1 + tf.exp(self.score))

    # def computePv(self, v, ul):
    #     pv = 1.0
    #     for u in ul:
    #         p_uv = tf.rsqrt(tf.reduce_sum(tf.square(tf.subtract(v, u))))
    #         pv = pv * (1 - p_uv)
    #     p_v = 1 - pv
    #     return p_v