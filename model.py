import tensorflow as tf

class PosSumWord():
    def __init__(self, 
                 batch_size, 
                 vocab_size,
                 pos_size, 
                 embedding_size,
                 sample_num,
                 learning_rate):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.pos_size = pos_size
        self.embedding_size = embedding_size
        self.sample_num = sample_num
        self.learning_rate = learning_rate

    def build_graph(self, default_name='PosSumWord'):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope(None, default_name=default_name):
                self.inputs = tf.placeholder(tf.int32, shape=[self.batch_size], name='inputs')
                self.targets = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='targets')
                self.words_matrix = [tf.placeholder(tf.int32, shape=None, name=f'words_matrix{i}') for i in range(self.batch_size)]
                
                self.pos_embeddings = tf.get_variable(name='pos_embeddings',
                                                      shape=[self.pos_size, self.embedding_size],
                                                      initializer=tf.contrib.layers.xavier_initializer())
                # scale
                self.pos_embeddings *= (self.embedding_size**0.5)

                pos_summed_vectors = []
                for i in range(self.batch_size):
                    lookuped = tf.nn.embedding_lookup(self.pos_embeddings, self.words_matrix[i])
                    # axis=0 ?
                    pos_summed = tf.reduce_sum(lookuped, axis=0)
                    pos_summed_vectors.append(pos_summed)
                
                word_embeddings = tf.stack(pos_summed_vectors)

                nce_weights = tf.get_variable(name='nce_weights',
                                              shape=[self.vocab_size, self.embedding_size],
                                              initializer=tf.contrib.layers.xavier_initializer())
                nce_biases =tf.Variable(tf.zeros([self.vocab_size]), name='nce_biases')

                with tf.variable_scope('loss'):
                    self.loss = tf.reduce_mean(
                                    tf.nn.nce_loss(weights=nce_weights,
                                                   biases=nce_biases,
                                                   labels=self.targets,
                                                   inputs=word_embeddings,
                                                   num_sampled=self.sample_num,
                                                   num_classes=self.vocab_size))
                
                with tf.variable_scope('optimizer'):
                    self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)