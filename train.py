import numpy as np
import tensorflow as tf
import collections
from konlpy.tag import Twitter
import argparse
import re
import math
import random

'''
    Step 1 : Parse Arguments.
'''
parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="input text file for training: one sentence per line")
parser.add_argument("--embedding_size", type=int, help="embedding vector size (default=150)", default=150)
parser.add_argument("--window_size", type=int, help="window size (default=5)", default=5)
parser.add_argument("--min_count", type=int, help="minimal number of word occurences (default=5)", default=5)
parser.add_argument("--num_sampled", type=int, help="number of negatives sampled (default=50)", default=50)
parser.add_argument("--learning_rate", type=float, help="learning rate (default=1.0)", default=1.0)
parser.add_argument("--sampling_rate", type=int, help="rate for subsampling frequent words (default=0.0001)", default=0.0001)
parser.add_argument("--epochs", type=int, help="number of epochs (default=3)", default=3)
parser.add_argument("--batch_size", type=int, help="batch size (default=150)", default=150)

args = parser.parse_args()


data, word_dict, word_reverse_dict, pos_dict, pos_reverse_dict, word_to_pos_dict \
        = build_dataset(args.input, args.min_count, args.sampling_rate)

vocabulary_size = len(word_dict)
pos_size = len(pos_dict)
num_sentences = len(data)

print("number of sentences :", num_sentences)
print("vocabulary size :", vocabulary_size)
print("pos size :", pos_size)

num_iterations = input_li_size // batch_size
print("number of iterations for each epoch :", num_iterations)
epochs = args.epochs
num_steps = num_iterations * epochs + 1

# keyed vector 이용하려고 이렇게 하는 거 같음.
# Function to save vectors.
def save_model(pos_list, embeddings, file_name):
    with open(file_name, 'w') as f:
        f.write(str(len(pos_list)))
        f.write(" ")
        f.write(str(embedding_size))
        f.write("\n")
        for i in range(len(pos_list)):
            pos = pos_list[i]
            f.write(str(pos).replace("', '", "','") + " ")
            f.write(' '.join(map(str, embeddings[i])))
            f.write("\n")

with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized - Tensorflow")

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(step, batch_size)

        word_list = []
        for word in batch_inputs:
            word_list.append(word_to_pos_dict[word])

        feed_dict = {}
        for i in range(batch_size):
            feed_dict[words_matrix[i]] = word_list[i]
        feed_dict[train_inputs] = batch_inputs
        feed_dict[train_labels] = batch_labels

        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        if step % 20000 == 0:
            pos_embed = pos_embeddings.eval()

            # Print nearest words
            sim = similarity.eval()
            for i in range(valid_size):
                valid_pos = pos_reverse_dict[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % str(valid_pos)
                for k in range(top_k):
                    close_word = pos_reverse_dict[nearest[k]]
                    log_str = '%s %s,' % (log_str, str(close_word))
                print(log_str)

    # Save vectors
    save_model(pos_li, pos_embeddings.eval(), "pos.vec")












