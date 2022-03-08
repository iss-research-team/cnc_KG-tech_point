#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/1 上午10:59
# @Author  : liu yuhan
# @FileName: ner_utils.py
# @Software: PyCharm
import numpy as np
import torch
import json
import pickle
import os
from transformers import BertTokenizer

from tqdm import tqdm


def not_empty(s):
    return s and s.strip()


def get_node_dict(word_list, label_list):
    """
    获取词语和词语对应的类
    :param word_list:
    :param label_list:
    :return:
    """
    words_dict = dict()
    words = []
    labels = []
    for word, label in zip(word_list, label_list):
        if label == 'O':
            if not words:
                continue
            words_dict[' '.join(words)] = labels[0]
            words = []
            labels = []
        else:
            words.append(word)
            labels.append(label)
    return words_dict


def make_data(data_list, tag_dict, max_len, num_class):
    data_size = len(data_list)

    token_list = []
    segment_list = []
    mask_list = []
    lens_list = []
    target_list = []

    # 通过词典导入分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # token_ids & target_ids
    for value in tqdm(data_list):
        # cut
        sentence, label_list = value
        node_dict = get_node_dict(sentence, label_list)
        sentence_trans = tokenizer(' '.join(sentence), truncation=True, max_length=max_len)
        token = sentence_trans['input_ids']
        mask = sentence_trans['attention_mask']
        segment = sentence_trans['token_type_ids']

        token_trans = ' ' + ' '.join([str(ids) for ids in token]) + ' '
        target = [0 for _ in range(len(token))]

        for node, label in node_dict.items():
            node_ids = tokenizer.encode(node, truncation=True, max_length=max_len)[1:-1]
            node_ids_trans = ' ' + ' '.join([str(ids) for ids in node_ids]) + ' '
            # 开始标记
            bit = 0
            for _ in range(token_trans.count(node_ids_trans)):
                bit = token_trans.find(node_ids_trans, bit)
                start_bit = token_trans[:bit].count(' ')
                target[start_bit] = tag_dict[label]
                for ids in range(start_bit + 1, start_bit + len(node_ids)):
                    target[ids] = tag_dict[label] + num_class

        token_list.append(token)
        segment_list.append(segment)
        mask_list.append(mask)
        target_list.append(target)
        lens_list.append(len(token))

    # 补pad
    for i in range(data_size):
        token_list[i].extend([0] * (max_len - lens_list[i]))
        segment_list[i].extend([0] * (max_len - lens_list[i]))
        mask_list[i].extend([0] * (max_len - lens_list[i]))
        target_list[i].extend([0] * (max_len - lens_list[i]))

    return torch.LongTensor(token_list), torch.LongTensor(segment_list), torch.LongTensor(mask_list), \
           torch.LongTensor(target_list), torch.LongTensor(lens_list)


def get_accuracy(pre_tenser, tag_tenser, lens_list):
    acc_list = []
    for p_t, t_t, length in zip(pre_tenser, tag_tenser, lens_list):
        # print(p_t)
        # print(t_t)
        num_tag = 0
        num_tag_acc = 0
        for bit in range(length):
            if t_t[bit] == 0:
                continue
            num_tag += 1
            if t_t[bit] == p_t[bit]:
                num_tag_acc += 1
        if num_tag == 0:
            continue
        acc_list.append(num_tag_acc / num_tag)
    return np.mean(acc_list)


def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == "__main__":
    # 生成数据+数据预处理
    max_len = 128
    with open('../data/3.expend/test_input/ner_dataset_public.json', 'r', encoding='UTF-8') as file:
        data_list_train = json.load(file)
    with open('../data/3.expend/test_input/ner_tag_dict_public.json', 'r', encoding='UTF-8') as file:
        tag_dict = json.load(file)
    make_data(data_list_train[:2000], tag_dict, max_len, num_class=8)
