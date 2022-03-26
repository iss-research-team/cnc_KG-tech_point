#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/3 上午9:34
# @Author  : liu yuhan
# @FileName: 2.topmine_add.py
# @Software: PyCharm

from collections import Counter
from tqdm import tqdm
import json
import sys


class TopmineAdd():
    def __init__(self, index_path, seg_path, min_freq):
        self.seg_path = seg_path
        self.index_path = index_path
        self.min_freq = min_freq
        # 构建索引
        self.index2word = dict()
        self.word2index = dict()
        self.get_index2word()
        self.word_dict_s = Counter()
        self.word_dict_m = Counter()
        self.get_frequency()

    def get_index2word(self):
        '''
        Returns a list of stopwords.
        '''
        f = open(self.index_path)
        words = []
        for line in f:
            words.append(line.rstrip())
        self.index2word = dict(zip([i for i in range(len(words))], words))
        self.word2index = dict(zip(words, [i for i in range(len(words))]))

    def add_base_keywords(self, keyword_base_path):
        '''
        专利尾部的关键词先引入进来
        :return:
        '''
        with open(keyword_base_path, 'r', encoding='UTF-8') as file:
            keywords = json.load(file)
        keywords = [keyword for keyword in keywords if keyword.count(' ') <= 5]
        # print('专利尾部的关键词总数：', len(keywords))
        count = 0
        for keyword in keywords:
            if ' ' in keyword:
                # 短语
                word_trans = []
                for word in keyword.split():
                    try:
                        word_trans.append(str(self.word2index[word]))
                    except KeyError:
                        break
                if len(word_trans) == keyword.count(' ') + 1:
                    word_trans = ' '.join(word_trans)
                    self.word_dict_m[word_trans] += 1
                    count += 1
            else:
                # 单词
                try:
                    word_trans = str(self.word2index[keyword])
                    self.word_dict_s[word_trans] += 1
                    count += 1
                except KeyError:
                    continue
        # print('其中可用的数量：', count)

    def get_frequency(self):
        '''
        计算词和词组的词频
        词组和单词貌似没有必要分开进行计算
        20211206
        词组和单词需要分开进行计算
        :return:
        '''
        docs = open(self.seg_path, 'r', encoding='UTF-8')
        for doc in docs:
            words = doc.split(', ')
            for word in words:
                if ' ' in word:
                    self.word_dict_m[word] += 1
                else:
                    self.word_dict_s[word] += 1

    def get_keywords(self, save_path_s, save_path_m):
        '''
        根据词频进行筛选
        这一阶段的词频筛选为第一阶段的词频筛选，选择一个小词频，用于术语的合并
        :return:
        '''
        # 存储词组
        count = 0
        file = open(save_path_m, 'w', encoding='UTF-8')
        for word, freq in self.word_dict_m.items():
            if freq > self.min_freq:
                word_trans = ' '.join([self.index2word[int(index)] for index in word.split()])
                file.write(word_trans + '\n')
                count += 1
        # print('num-keyword-m:', count)
        num_keyword_m = count
        # 存储单词
        count = 0
        file = open(save_path_s, 'w', encoding='UTF-8')
        for word, freq in self.word_dict_s.items():
            if freq > self.min_freq:
                word_trans = self.index2word[int(word)]
                file.write(word_trans + '\n')
                count += 1
        # print('num-keyword-s:', count)
        num_keyword_s = count
        return num_keyword_m, num_keyword_s


if __name__ == '__main__':
    label = 'literature'
    seg_path = '../data/1.keyword_get/topmine/partitioneddocs_' + label + '.txt'
    index_path = '../data/1.keyword_get/topmine/vocab_' + label + '.txt'
    keyword_base_path = '../data/1.keyword_get/cnc_keywords_base_' + label + '.txt'
    save_path_s = '../data/1.keyword_get/cnc_keywords_single_' + label + '.txt'
    save_path_m = '../data/1.keyword_get/cnc_keywords_multiple_' + label + '.txt'

    num_m_list = []
    num_s_list = []

    freq_list = [10 + i * 10 for i in range(100)]

    for min_freq in tqdm(freq_list):
        topmine_add = TopmineAdd(index_path, seg_path, min_freq)
        topmine_add.add_base_keywords(keyword_base_path)
        num_keyword_m, num_keyword_s = topmine_add.get_keywords(save_path_s, save_path_m)
        num_m_list.append(num_keyword_m)
        num_s_list.append(num_keyword_s)

    with open('../fig/1.keyword_get/keyword_num_' + label + '.json', 'w', encoding='UTF-8') as file:
        json.dump([freq_list, num_s_list, num_m_list], file)
