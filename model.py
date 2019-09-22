embedding_size = args.embedding_size
num_sampled = args.num_sampled
learning_rate = args.learning_rate

valid_size = 20     # Random set of words to evaluate similarity on.
valid_window = 200  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()

with graph.as_default():
    # Input data
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    words_matrix = [tf.placeholder(tf.int32, shape=None) for _ in range(batch_size)]
    vocabulary_matrix = [tf.placeholder(tf.int32, shape=(None)) for _ in range(vocabulary_size)]
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        pos_embeddings = tf.Variable(tf.random_uniform([pos_size, embedding_size], -1.0, 1.0), name='pos_embeddings')

        word_vec_list = []
        for i in range(batch_size):
            # words_vec을 pos_vec의 합으로 정의, axis = 0인 게 중요
            word_vec = tf.reduce_sum(tf.nn.embedding_lookup(pos_embeddings, words_matrix[i]), 0)
            word_vec_list.append(word_vec)
        # stack 왜 하지?
        # word_embeddigs trainable이든 아니든 variable로 해야 저장 가능
        word_embeddings = tf.stack(word_vec_list)

        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)), name='nce_weights'
        )
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name='nce_biases')

    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=word_embeddings,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    init = tf.global_variables_initializer()

    # Compute the cosine similarity between minibatch exaples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(pos_embeddings), 1, keep_dims=True))
    normalized_embeddings = pos_embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)