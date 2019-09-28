import argparse
import random
import os 

import numpy as np
import tensorflow as tf
from konlpy.tag import Mecab

from preprocessor import Loader
from model import PosSumWord

def get_feed_dict(model, loader, inputs, targets):
    feed_dict = {}    
    for placeholder, input_ in zip(model.words_matrix, inputs):
        feed_dict[placeholder] = loader.word_idx_to_pos_idx_list[input_]

    feed_dict.update(
                    {
                        model.inputs: inputs,
                        model.targets: targets
                    }
                )    

    return feed_dict

def save_embeddings(idx_pos, embeddings, save_path, project_name, max_step):
    embeddings_name = f"{project_name}_max_step{max_step}_morpheme_embeddings.kv"
    embeddings_save_path = os.path.join(save_path, embeddings_name)
    evaled_embeddings = embeddings.eval()

    with open(embeddings_save_path, "w") as out:
        out.write(f"{len(idx_pos)} {evaled_embeddings.shape[1]}\n")
        
        for idx, pos in idx_pos.items():
            out.write(" ".join([pos] + list(evaled_embeddings[idx].astype(str))))
            out.write("\n")
    
    print(f"save {embeddings_name} in {save_path}")

def main(args):
    loader = Loader()

    loader.load(args.preprocessed_path)
    pos_sum_word_model = PosSumWord(batch_size=args.batch_size,
                                    vocab_size=len(loader.word_idx),
                                    pos_size=len(loader.pos_idx),
                                    embedding_size=args.embedding_size,
                                    sample_num=args.num_sampled,
                                    learning_rate=args.learning_rate)
    
    save_dir = os.path.join(args.save_path, 'checkpoints')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    project_name = ("PosSumWord_"
                    f"batch{args.batch_size}_"
                    f"embed{args.embedding_size}_"
                    f"window{args.window_size}_"
                    f"min_cnt{args.min_count}_"
                    f"num_sampled{args.num_sampled}_"
                    f"lr{args.learning_rate}")
    ckpt_save_path = os.path.join(save_dir, f"{project_name}.ckpt")

    gen = loader.get_generator(args.batch_size)
    pos_sum_word_model.build_graph()                                
    
    save_step = args.max_step // args.max_to_keep
    print_step = args.max_step // 100
    latest_step = 0

    with tf.Session(graph=pos_sum_word_model.graph) as session:
        init = tf.global_variables_initializer()
        session.run(init)
        saver = tf.train.Saver(max_to_keep=args.max_to_keep)

        if args.is_load:
            latest_ckpt = tf.train.latest_checkpoint(save_dir)
            assert latest_ckpt != None, "latest_ckpt == None"
            latest_step = int(latest_ckpt.split("-")[-1])
            saver.restore(session, latest_ckpt)
            print(f"{latest_ckpt} loaded")

        avg_loss = 0
        for step in range(1, args.max_step+1):
            inputs, targets = next(gen)
            targets = np.expand_dims(targets, 1)
            feed_dict = get_feed_dict(pos_sum_word_model, loader, inputs, targets)

            _, loss = session.run([pos_sum_word_model.optimizer, pos_sum_word_model.loss], 
                                   feed_dict=feed_dict)
            avg_loss += loss

            if (step) % print_step == 0:
                if step > 0:
                    avg_loss /= print_step
                print("Batch Average loss at step ", step+latest_step, ": ", avg_loss)

            if (step) % save_step == 0:
                saver.save(session, ckpt_save_path, global_step=step+latest_step)

        # Save vectors
        save_embeddings(idx_pos=loader.idx_pos, 
                        embeddings=pos_sum_word_model.pos_embeddings, 
                        save_path=args.save_path,
                        project_name=project_name,
                        max_step=args.max_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("PosSumWordTrain")
    parser.add_argument("--preprocessed_path", type=str, help="path for preprocessed data")
    parser.add_argument("--save_path", type=str, help="path for saving model ckpt and vector", default='/tmp')
    parser.add_argument("--embedding_size", type=int, help="embedding vector size (default=128)", default=128)
    parser.add_argument("--window_size", type=int, help="window size (default=3)", default=3)
    parser.add_argument("--min_count", type=int, help="minimal number of word occurences (default=1)", default=1)
    parser.add_argument("--num_sampled", type=int, help="number of negatives sampled (default=64)", default=64)
    parser.add_argument("--learning_rate", type=float, help="learning rate (default=0.0001)", default=0.0001)
    parser.add_argument("--sampling_rate", type=int, help="rate for subsampling frequent words (default=0.0001)", default=0.0001)
    parser.add_argument("--sampling_threshold", type=float, help="threshold for sampling probability", default=0.9)
    parser.add_argument("--max_step", type=int, help="max train steps")
    parser.add_argument("--batch_size", type=int, help="batch size (default=128)", default=128)
    parser.add_argument("--max_to_keep", type=int, help="max latest ckpt num to save", default=10)
    parser.add_argument("--is_load", action='store_true', help="load model ckpt")

    args = parser.parse_args()

    main(args)