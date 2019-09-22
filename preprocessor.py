import os
import re
import random
from datetime import datetime

import numpy as np
from konlpy.tag import Mecab
from bounter import bounter

class CommonVar():
    def __init__(self):
        self.word_idx, self.idx_word = {}, {}
        self.pos_idx, self.idx_pos = {}, {}
        self.word_idx_to_pos_idx_list = {}
        self.inputs, self.targets = [], []
        self.var_names = ['word_idx', 'idx_word', 'pos_idx', 'idx_pos', 
                          'word_idx_to_pos_idx_list', 'inputs', 'targets']

class Preprocessor(CommonVar):
    def __init__(self):
        super().__init__()
        self.unk_token = '<UNK>'
        self.pos_delimeter = '/'

    def preprocess(self, path, min_cnt=5, sampling_rate=0.0001, window_size=5, sampling_threshold=0.9):
        sentences = self.get_sentences(path)
        
        word_cnts, valid_words, n_total_words = self.get_word_cnts_gt_min_cnt(sentences, min_cnt)
        sub_sampled_sentences = self.sub_sampling(sentences, word_cnts, sampling_rate, n_total_words, sampling_threshold)

        for i, word in enumerate(valid_words):
            self.word_idx[word] = i
            self.idx_word[i] = word
        
        self.set_word_idx_to_pos_idx_list(valid_words)
        self.get_inputs_targets(sub_sampled_sentences, window_size)
        
    def save(self, path='/tmp/'):
        save_dir = f"kor2vec_preprocessed_{datetime.now().strftime('%Y-%m-%d-%H:%M')}"
        os.makedirs(os.path.join(path, save_dir))
        
        for var_name in self.var_names:
            save_path = os.path.join(path, save_dir, f'{var_name}.npy')
            np.save(save_path, getattr(self, var_name))
            print(var_name, 'saved in', save_path)

    def get_sentences(self, path):
        sentences = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                cleaned = re.sub("[^ㄱ-힣a-zA-Z0-9 .]", '', line).strip()
                if cleaned:
                    dot_splited = cleaned.split('.')
                    for str_sentence in dot_splited:
                        splited = str_sentence.split()
                        if splited:
                            sentences.append(splited)
        return sentences

    def get_word_cnts_gt_min_cnt(self, sentences, min_cnt):
        flattened = [word for sentence in sentences for word in sentence]
        counter = bounter(size_mb=4096)
        counter.update(flattened)
        
        array_cnt = np.array(list(counter.iteritems()))
        array = array_cnt[:, 0]
        cnts = array_cnt[:, 1]
        cnts = cnts.astype(int)

        valid = np.where(cnts > min_cnt)
        valid_array = array[valid]
        valid_cnts = cnts[valid]
        word_cnts = list(zip(valid_array, valid_cnts))
        
        return word_cnts, valid_array, len(flattened)

    def sub_sampling(self, sentences, word_cnts, sampling_rate, n_total_words, sampling_threshold):
        word_prob = {}
        for word, cnt in word_cnts:
            freq = cnt / n_total_words
            prob = 1 - np.sqrt(sampling_rate / freq)
            prob = max(0, prob)
            word_prob[word] = prob
        
        sub_sampled_sentences = []
        for sentence in sentences:
            sampled_sentence = []
            for word in sentence:
                if word not in word_prob:
                    sampled_sentence.append(self.unk_token)
                else:
                    if sampling_threshold > word_prob[word]:
                        sampled_sentence.append(word)
            sub_sampled_sentences.append(sampled_sentence)
        
        return sub_sampled_sentences            

    def set_word_idx_to_pos_idx_list(self, valid_words):
        mecab = Mecab()
        idx = 0

        for word in valid_words:
            pos_list = mecab.pos(word)
            pos_idx_list = []
            for pos in pos_list:
                joined = self.pos_delimeter.join(pos)
                if joined not in self.pos_idx:
                    self.pos_idx[joined] = idx
                    self.idx_pos[idx] = joined
                    idx += 1
                pos_idx_list.append(self.pos_idx[joined])
            self.word_idx_to_pos_idx_list[self.word_idx[word]] = pos_idx_list

    def get_inputs_targets(self, sub_sampled_sentences, window_size):
        for sentence in sub_sampled_sentences:
            for i in range(len(sentence)):
                start = max(0, i - window_size)
                end = min(len(sentence), i + window_size + 1)
                for j in range(start, end):
                    if i != j and\
                        sentence[i] in self.word_idx and\
                        sentence[j] in self.word_idx:
                        self.inputs.append(self.word_idx[sentence[i]])
                        self.targets.append(self.word_idx[sentence[j]])    

def generate_batch(iter, batch_size):
    index = (iter % (input_li_size//batch_size)) * batch_size
    batch_input = input_li[index:index+batch_size]
    batch_output_li = output_li[index:index+batch_size]
    batch_output = [[i] for i in batch_output_li]

    return np.array(batch_input), np.array(batch_output)