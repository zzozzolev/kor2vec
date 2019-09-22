import re
import math

import numpy as np
from konlpy.tag import Mecab
from bounter import bounter

class CommonVar():
    def __init__(self):
        self.word_idx, self.idx_word = {}, {}
        self.pos_idx, self.idx_pos = {}, {}
        self.word_idx_to_pos_id_list = {}
        self.inputs, self.targets = [], []


class Preprocessor(CommonVar):
    def __init__(self):
        super().__init__()
    
    def preprocess(self, path, min_count, sampling_rate, window_size):
        pass

    def save():
        pass

    def get_sentences(self, path):
        pass

    def get_word_freq(self, sentences):
        pass
    
    def set_idx_dict(self, value, value_idx_dict, idx_value_dict):
        pass

    def sub_sampling(self, word_freq, sampling_rate):
        pass

    def set_word_idx_to_pos_id_list(self, sub_sampled_sentences):
        pass

    def get_inputs_targets(self, sub_sampled_sentences, window_size):
        pass
    

def build_dataset(train_text, min_count, sampling_rate):
    words = list()
    with open(train_text, 'r') as f:
        lines = f.readlines()
        for line in lines:
            sentence = re.sub(r"[^ㄱ-힣a-zA-Z0-9]+", ' ', line).strip().split()
            if sentence:
                words.append(sentence)

    # 단어 빈도 수 세기
    word_counter = [['UNK', -1]]
    word_counter.extend(collections.Counter([word for sentence in words for word in sentence]).most_common())
    word_counter = [item for item in word_counter if item[1] >= min_count or item[0] == 'UNK']

    # key가 단어이고 value가 idx인 빈도 수 사전 만들기
    word_dict = dict()
    for word, count in word_counter:
        word_dict[word] = len(word_dict)
    word_reverse_dict = dict(zip(word_dict.values(), word_dict.keys()))

    # key가 단어의 idx이고 value가 pos list인 딕셔너리 만들기
    word_to_pos_li = dict()
    pos_list = list()
    twitter = Twitter()
    for w in word_dict:
        w_pos_li = list()
        for pos in twitter.pos(w, norm=True):
            w_pos_li.append(pos)

        word_to_pos_li[word_dict[w]] = w_pos_li
        pos_list += w_pos_li

    # 형태소 빈도 수 세기
    pos_counter = collections.Counter(pos_list).most_common()

    # key가 형태소이고 value가 idx인 딕셔너리 만들기
    pos_dict = dict()
    for pos, _ in pos_counter:
        pos_dict[pos] = len(pos_dict)

    pos_reverse_dict = dict(zip(pos_dict.values(), pos_dict.keys()))

    word_to_pos_dict = dict()

    # key가 word의 idx이고 value가 형태소의 idx의 list인 딕셔너리 만들기
    for word_id, pos_li in word_to_pos_li.items():
        pos_id_li = list()
        for pos in pos_li:
            pos_id_li.append(pos_dict[pos])
        word_to_pos_dict[word_id] = pos_id_li

    # sentence string에서 idx로 바꾸기
    data = list()
    unk_count = 0
    for sentence in words:
        s = list()
        for word in sentence:
            if word in word_dict:
                index = word_dict[word]
            else:
                index = word_dict['UNK']
                unk_count += 1
            s.append(index)
        data.append(s)
    # UNK 빈도 수 값 치환
    word_counter[0][1] = max(1, unk_count)

    data = sub_sampling(data, word_counter, word_dict, sampling_rate)

    return data, word_dict, word_reverse_dict, pos_dict, pos_reverse_dict, word_to_pos_dict


# Sub-sampling frequent words according to sampling_rate
def sub_sampling(data, word_counter, word_dict, sampling_rate):
    total_words = sum([len(sentence) for sentence in data])
    prob_dict = dict()
    # 자주 등장하는 단어일 수록 p가 커짐
    for word, count in word_counter:
        f = count / total_words
        p = max(0, 1 - math.sqrt(sampling_rate / f))
        prob_dict[word_dict[word]] = p

    # random보다 크면 sentence에서 해당 단어를 버림    
    new_data = list()
    for sentence in data:
        s = list()
        for word in sentence:
            prob = prob_dict[word]
            if random.random() > prob:
                s.append(word)
        new_data.append(s)

    return new_data

pos_li = []
for key in sorted(pos_reverse_dict):
    pos_li.append(pos_reverse_dict[key])

window_size = args.window_size
batch_size = args.batch_size

def generate_input_output_list(data, window_size):
    input_li = list()
    output_li = list()
    for sentence in data:
        for i in range(len(sentence)):
            # IndexError 방지
            for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                if i != j:
                    if sentence[i]!=word_dict['UNK'] and sentence[j]!=word_dict['UNK']:
                        input_li.append(sentence[i])
                        output_li.append(sentence[j])
    return input_li, output_li

input_li, output_li = generate_input_output_list(data, window_size)
input_li_size = len(input_li)


def generate_batch(iter, batch_size):
    index = (iter % (input_li_size//batch_size)) * batch_size
    batch_input = input_li[index:index+batch_size]
    batch_output_li = output_li[index:index+batch_size]
    batch_output = [[i] for i in batch_output_li]

    return np.array(batch_input), np.array(batch_output)