import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net,bilinear


class Model(object):
    def __init__(self, config, batch, word_mat=None, trainable=True, opt=True):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.c, self.q, self.alternatives, self.y, self.qa_id = batch.get_next()
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False)

        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.alter_mask = tf.cast(self.alternatives, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
        self.alter_len = tf.reduce_sum(tf.cast(self.alter_mask, tf.int32), axis=1)

        if opt:
            N = config.batch_size
            self.c_maxlen = tf.reduce_max(self.c_len)
            self.q_maxlen = tf.reduce_max(self.q_len)
            self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
            self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
            self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
            self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
            self.y = tf.slice(self.y, [0, 0], [N, 3])

        else:
            self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

        self.ready()

        if trainable:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.lr, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

    def ready(self):
        config = self.config
        N, PL, QL, d= config.batch_size, self.c_maxlen, self.q_maxlen, config.hidden,
        gru = cudnn_gru if config.use_cudnn else native_gru

        # 词向量层
        with tf.variable_scope("emb"):
            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                # q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)
                # alter_emb = tf.nn.embedding_lookup(self.word_mat, self.alternatives)

        # 编码层
        with tf.variable_scope("encoding"):
            rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=c_emb.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            # [batch, c_size, hidden_size]
            c = rnn(c_emb, seq_len=self.c_len)
            # q = rnn(q_emb, seq_len=self.q_len)
            # alter = rnn(alter_emb, seq_len=self.alter_len)

        # with tf.variable_scope("attention"):
        #     qc_att = dot_attention(c, q, mask=self.q_mask, hidden=d,
        #                            keep_prob=config.keep_prob, is_train=self.is_train)
        #     rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
        #     ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
        #     att = rnn(qc_att, seq_len=self.c_len)
        #
        # with tf.variable_scope("match"):
        #     self_att = dot_attention(
        #         att, att, mask=self.c_mask, hidden=d, keep_prob=config.keep_prob, is_train=self.is_train)
        #     rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=self_att.get_shape(
        #     ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
        #     match = rnn(self_att, seq_len=self.c_len)

        with tf.variable_scope("predict"):
            # evidence for passage
            # r_p = tf.reduce_mean(match, axis=1)
            # # 在1维上插入一个长度为1的axis
            # r_p = tf.expand_dims(r_p, 1)
            # # [batch, 1 , 3]
            # logits = bilinear(r_p, alter)
            dense1 = tf.layers.dense(c, units=80, activation=tf.nn.relu)
            dense1 = tf.reduce_mean(dense1, axis=1)
            logits = tf.layers.dense(dense1, units=3, activation=tf.nn.tanh)
            # logits = tf.reshape(logits, [N, 3])
            self.yp = tf.argmax(tf.nn.softmax(logits), axis=1)

            losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=tf.stop_gradient(self.y)
                )
            self.loss = tf.reduce_mean(losses)
        # with tf.variable_scope("compute_z"):
        #     v1 = tf.reduce_sum(c, axis=1)
        #     v2 = tf.reduce_sum(q, axis=1)
        #     v_all = tf.concat([v1, v2], axis=1)
        #
        #     v_all = dropout(v_all, keep_prob=config.keep_prob, is_train=self.is_train)
        #     # MPL
        #     z = tf.layers.dense(tf.layers.dense(v_all, units=80, activation=tf.nn.relu),
        #                         units=3, activation=tf.nn.tanh)
        # with tf.variable_scope("predict"):
        #     losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        #         logits=z, labels=tf.stop_gradient(self.y)
        #     )
        #     self.loss = tf.reduce_mean(losses)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step