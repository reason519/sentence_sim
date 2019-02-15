import tensorflow as tf

class SentSimRNN(object):
    def __init__(self,is_training,batch_size,max_sent_size,hidden_units,learning_rate,init_emb=None):
        self.is_training=is_training
        self.batch_size=batch_size
        self.max_sent_size=max_sent_size
        self.hidden_size=hidden_units
        self.init_emb=init_emb

        self.learning_rate=learning_rate

        self.keep_prob = 1.0

        self.max_gradient_norm = 1.25
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        decay_steps = 50
        decay_rate = 0.96
        self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps, decay_rate,
                                                        staircase=True)

        self.session = tf.Session()
        self._build_input()
        self._model()

    def _build_input(self):
        self.x_sent1 = tf.placeholder(tf.int32, shape=(None, self.max_sent_size), name='x_sent1')
        self.x_sent2 = tf.placeholder(tf.int32, shape=(None, self.max_sent_size), name='x_sent2')
        self.x_sent1_len = tf.placeholder(tf.int32, shape=(None), name='x_sent1_len')
        self.x_sent2_len = tf.placeholder(tf.int32, shape=(None), name='x_sent2_len')
        if self.is_training:
            self.y_target=tf.placeholder(tf.float32,shape=(None),name='y_target')

    def _model(self):
        with tf.variable_scope('emb'):
            if self.init_emb is not None:
                self.word_embeddings = tf.get_variable('word_embeddings',initializer=self.init_emb, trainable=False)

          #      self.word_embeddings=tf.get_variable('word_embeddings',self.init_emb.shape,initializer=tf.contrib.layers.xavier_initializer(),trainable=True,dtype=tf.float32)
          #     self.word_embeddings = tf.get_variable('word_embeddings', self.init_emb.shape,
          #                                            initializer=tf.random_normal_initializer(), trainable=True,
          #                                            dtype=tf.float32)
        x_sent1_emb=tf.nn.embedding_lookup(self.word_embeddings,self.x_sent1)
        x_sent2_emb=tf.nn.embedding_lookup(self.word_embeddings,self.x_sent2)


        with tf.variable_scope('lstm'):
            lstm=tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
            output1,state1=tf.nn.dynamic_rnn(lstm,x_sent1_emb,self.x_sent1_len,dtype=tf.float32)

            output2, state2 = tf.nn.dynamic_rnn(lstm, x_sent2_emb,self.x_sent2_len, dtype=tf.float32)
            #manhattan distance
            diff=tf.reduce_sum(tf.abs(state1.h-state2.h),axis=1)

            self.manhattan_distance=tf.clip_by_value(tf.exp(-1.0 * diff), 1e-7, 1.0 - 1e-7)
            self.loss=tf.square(tf.subtract(self.manhattan_distance,tf.clip_by_value((self.y_target - 1.0) / 4.0, 1e-7, 1.0 - 1e-7)))
            # self.manhattan_distance = tf.clip_by_value(tf.exp(-1.0 * diff), 1.0, 5.0)
            # self.loss = tf.square(  tf.subtract(self.manhattan_distance, self.y_target ))

            #cosin distance
            # leftNorm=tf.nn.l2_normalize(state1.h)
            # rightNorm = tf.nn.l2_normalize(state2.h)
            # self.manhattan_distance = tf.exp(tf.reduce_sum(tf.multiply(rightNorm,leftNorm), axis=1))



            if self.is_training:
                self.cost = tf.reduce_mean(self.loss)
                self.mse = tf.reduce_mean(tf.square(tf.subtract(self.manhattan_distance * 4.0 + 1.0, self.y_target)))
               # self.mse=tf.losses.mean_squared_error(self.manhattan_distance,self.y_target) #+tf.losses.get_regularization_loss()

                optimizer = tf.train.AdamOptimizer(self.learning_rate)
             #   optimizer=tf.train.AdadeltaOptimizer(self.learning_rate)
                trainable_params = tf.trainable_variables()
                # Compute gradients of loss w.r.t. all trainable variables
                gradients = tf.gradients(self.cost, trainable_params)
                # Clip gradients by a given maximum_gradient_norm
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.updates = optimizer.apply_gradients(
                    zip(clip_gradients, trainable_params), global_step=self.global_step)
                self.session.run(tf.global_variables_initializer())

    def train(self,x_sent1,x_sent2,y_target,x_sent1_len,x_sent2_len,train_cond=True):
        input_feed={
            self.x_sent1:x_sent1,self.x_sent2:x_sent2,self.y_target:y_target,self.x_sent1_len:x_sent1_len,
            self.x_sent2_len:x_sent2_len
        }
        if train_cond:
            _, loss, pred = self.session.run([self.updates,self.mse, self.manhattan_distance], feed_dict=input_feed)
        else:
            loss, pred = self.session.run([self.mse, self.manhattan_distance], feed_dict=input_feed)
        return loss,pred