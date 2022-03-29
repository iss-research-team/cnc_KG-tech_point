#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 下午4:58
# @Author  : liu yuhan
# @FileName: keyword_search.py
# @Software: PyCharm

from collections import Counter
from tqdm import tqdm
import json


class Search():
    def __init__(self, index_path, seg_path):
        self.seg_path = seg_path
        self.index_path = index_path
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
        print('专利尾部的关键词总数：', len(keywords))
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
        print('其中可用的数量：', count)

    def get_frequency(self):
        '''
        计算词和词组的词频
        词组和单词貌似没有必要分开进行计算
        20211206
        词组和单词需要分开进行计算
        :return:
        '''
        docs = open(self.seg_path, 'r', encoding='UTF-8')
        for doc in tqdm(docs):
            words = doc.split(', ')
            for word in words:
                if ' ' in word:
                    self.word_dict_m[word] += 1
                else:
                    self.word_dict_s[word] += 1

    def search(self, word):
        if ' ' in word:
            word_trans = ' '.join([str(self.word2index[word]) for word in word.split()])
            freq = self.word_dict_m[word_trans]
        else:
            word_trans = str(self.word2index[word])
            freq = self.word_dict_s[word_trans]

        print('freq:', freq)


if __name__ == '__main__':
    label = 'literature'
    seg_path = '../data/1.keyword_get/topmine/partitioneddocs_' + label + '.txt'
    index_path = '../data/1.keyword_get/topmine/vocab_' + label + '.txt'
    keyword_base_path = '../data/1.keyword_get/keywords/cnc_keywords_base_' + label + '.txt'
    save_path_s = '../data/1.keyword_get/keywords/cnc_keywords_single_' + label + '.txt'
    save_path_m = '../data/1.keyword_get/keywords/cnc_keywords_multiple_' + label + '.txt'

    search = Search(index_path, seg_path)
    search.add_base_keywords(keyword_base_path)
    search.search('five axis')
    search.search('error compensation')
    # search.search('tool path plan')
