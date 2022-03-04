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


def make_data(data_list, tag_dict, max_len):
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
        token, target = value
        token_trans, target_trans = [], []
        for word, tag in zip(token, target):
            token_trans.append(tokenizer.convert_tokens_to_ids(word.lower()))
            target_trans.append(tag_dict[tag])

        token_trans = [101] + token_trans + [102]
        target_trans = [0] + target_trans + [0]

        mask = [1] * len(token_trans)
        segment = [0] * len(token_trans)

        token_list.append(token_trans)
        segment_list.append(segment)
        mask_list.append(mask)
        target_list.append(target_trans)
        lens_list.append(len(token_trans))

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
        print(p_t)
        print(t_t)
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
