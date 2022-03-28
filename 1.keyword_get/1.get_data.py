#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 下午4:46
# @Author  : liu yuhan
# @FileName: 1.get_data.py
# @Software: PyCharm

"""
晚上解决两个问题：
1.连字符
2.数字替换
"""

import json
import re
import spacy
from tqdm import tqdm
import sys


def not_empty(s):
    return s and s.strip()


class DataProcess:
    """
    类说明：
    对文本的进一步的预处理
    """

    def __init__(self, label, doc_path, doc_index_path, keyword_path):
        self.label = label
        print('label:', label)
        self.doc_path = doc_path
        self.doc_index_path = doc_index_path
        self.keyword_path = keyword_path

        # 处理过程中用到的正则表达式子
        self.pattern_patent = re.compile(r'\(.*?\)')
        self.pattern_lite = re.compile(r'\[.*?]')
        self.pattern_sp = re.compile(r'\s\W+\s')
        self.pattern_number = re.compile(r'\s[0-9]\s')
        self.num_trans_dict = {'0': 'zero',
                               '1': 'one',
                               '2': 'two',
                               '3': 'three',
                               '4': 'four',
                               '5': 'five',
                               '6': 'six',
                               '7': 'seven',
                               '8': 'eight',
                               '9': 'nine'}

        # 词形还原的工具
        self.lemma_tool = spacy.load("en_core_web_sm")  # load package "en_core_web_sm"
        # 载入停词表
        self.get_stopwords()
        # 用于获取专利尾部的关键词和论文的关键词
        self.keywords = []

    def get_last_s(self, doc):
        """
        句子拆分，得到最后一个句子
        :param doc:
        :return:
        """
        bit = doc.rfind('.', 0, doc.rfind('.'))
        return doc[:bit + 1].strip(), doc[bit + 1:-1].strip()

    def get_stopwords(self):
        """
        Returns a list of stopwords.
        """
        f = open("stopwords.txt")
        stopwords = set()
        for line in f:
            stopwords.add(line.rstrip())
        self.stopwords = stopwords

    def number_trans(self, string):
        """
        补获文本中的数字，将其置换为英文
        :return:
        """
        string = ' ' + string + ' '
        number_list = re.findall(self.pattern_number, string)
        for number in number_list:
            string = string.replace(number, ' ' + self.num_trans_dict[number.strip()] + ' ')
        return string.strip()

    def str_deal(self, string):
        """
        获取句子或者词语的词形还原结果
        :return:
        """
        string = self.lemma_tool(string)
        string = ' '.join([token.lemma_ for token in string])
        # 恢复连字符
        string = string.replace(' - ', '-')
        string = ' '.join([word for word in string.split() if word.lower() not in self.stopwords])
        # 停词防止误伤后，去除连字符
        string = string.replace('-', ' ')
        string = self.number_trans(string)
        string = string.lower()
        return ' ' + string + ' '

    def doc_deal(self, doc):
        """
        获取句子或者词语的词形还原结果
        :return:
        """
        doc = self.str_deal(doc)
        doc = re.sub(self.pattern_sp, ' ', doc)
        return doc.strip() + '. '

    def keyword_deal(self, keyword):
        """
        获取句子或者词语的词形还原结果
        :return:
        """
        keyword = re.sub(r"[,.;:!?|/\\]", ' ', keyword)
        keyword = self.str_deal(keyword)
        return keyword.strip()

    def get_keywords_patent(self, keyword_temper):
        """
        最后挣扎一次，按照设计的六种分割方法
        :param keywords_temper:
        :return:
        """
        keyword_temper = keyword_temper \
            .replace('USES', '') \
            .replace('USE', '') \
            .replace('ADVANTAGE', '')

        if re.search('[0-9]+[A-Za-z]', keyword_temper):
            # key123Key
            # key1, 3Key
            # key1-3Key
            # key1-3aKey
            # key1-3AKey
            pattern_keyword = re.compile('\d[a-z0-9-, ]*')
            keyword_temper = re.sub(pattern_keyword, ' | ', keyword_temper)
            keyword_temper = list(filter(not_empty, keyword_temper.split(' | ')))
            for i, keyword in enumerate(keyword_temper):
                # 多余的大写字母
                sp = re.search('[A-Z]+[A-Z]', keyword)
                if sp:
                    keyword_temper[i] = keyword[len(sp.group()) - 1:]
        elif re.search('[A-Z]+[A-Z][a-z]', keyword_temper):
            # keyAAKey
            # keyA, AKey
            # keyA-AKey
            sp_list = re.findall('[A-Z][0-9-, ]*[A-Z]+', keyword_temper)
            if sp_list:
                for sp in sp_list:
                    keyword_temper = keyword_temper.replace(sp[:-1], ' | ')
            keyword_temper = list(filter(not_empty, keyword_temper.split(' | ')))
        else:
            keyword_temper = []
        if keyword_temper:
            keyword_temper = [self.keyword_deal(keyword.strip()) for keyword in keyword_temper]

        return keyword_temper

    def get_keywords_lite(self, keyword_temper):
        """
        论文关键词的提取
        :param keywords_temper:
        :return:
        """
        keyword_temper = keyword_temper.split('; ')
        keyword_temper = [self.keyword_deal(keyword.strip()) for keyword in keyword_temper]

        return keyword_temper

    def doc_trans(self, doc):
        """
        通用的处理，初步的处理
        包括括号中的信息，引号等
        :param doc:
        :return:
        """
        doc = re.sub(r"['\"]", ' ', doc)
        # 需要解决括号的问题
        if self.label == 'patent':
            char_list = re.findall(self.pattern_patent, doc)
        else:
            char_list = re.findall(self.pattern_lite, doc)

        for char in char_list:
            # 规则可以继续叠加
            if char[1].isdigit() or \
                    'non-English language text' in char:
                doc = doc.replace(char, '')
        doc = doc.replace(' e.g.', ',').replace(' i.e.', ',')
        doc = doc.replace(' E.g.', ',').replace(' I.e.', ',')
        doc = doc.replace(' E.G.', ',').replace(' I.E.', ',')
        doc = ' '.join(doc.split())
        doc = re.sub(r"\.\W*\.", '.', doc)
        return doc.strip()

    def doc_trans_patent(self, doc):
        """
        针对专利数据
        对单一句子的处理
        :param doc:
        :return:
        """
        doc = self.doc_trans(doc)
        # 找到倒数两个句号，中间的话就是最后一句话
        doc_a, keywords_str = self.get_last_s(doc)
        # 修改这个判断的部分
        keywords_temper = self.get_keywords_patent(keywords_str)
        if keywords_temper:
            sentence_list = [self.doc_deal(sentence) for sentence in doc_a.split('. ')]
        else:
            sentence_list = [self.doc_deal(sentence) for sentence in doc.split('. ')]
        # 去除一些奇怪的连接符号
        return sentence_list, keywords_temper

    def doc_trans_lite(self, doc):
        """
        论文的数据预处理相对简单
        :param doc:
        :return:
        """
        doc = self.doc_trans(doc)
        sentence_list = [self.doc_deal(sentence) for sentence in doc.split('. ')]
        return sentence_list

    def get_date(self):
        """
        获取文本数据
        :return:
        """
        with open('../data/0.preprocessed_file/doc_dict_' + self.label + '.json', 'r', encoding='UTF-8') as file:
            data_dict = json.load(file)
        with open('../data/0.preprocessed_file/doc_' + self.label + '2index.json', 'r', encoding='UTF-8') as file:
            index_dict = json.load(file)
        # write
        f_write = open(self.doc_path, 'w', encoding='UTF-8')
        doc2index = dict()
        print('文本数量：', len(data_dict))
        count = 0
        for t, inf in tqdm(data_dict.items()):
            # 分流，专利处理
            sentence_list = []
            keyword = []
            if self.label == 'patent':
                doc = inf['doc']
                # 关键词来自于文本的最后一句话
                sentence_list, keyword = self.doc_trans_patent(doc)
            # 论文处理
            if self.label == 'literature':
                doc = t + ' ' + inf['doc']
                sentence_list = self.doc_trans_lite(doc)
                keyword = inf['key_words_author']
                keyword = self.get_keywords_lite(keyword)

            for sentence in sentence_list:
                f_write.write(sentence + '\n')
                doc2index[sentence] = index_dict[t]
            self.keywords += keyword
            # count += 1
            # if count == 1000:
            #     break

        f_write.close()
        with open(self.doc_index_path, 'w', encoding='UTF-8') as file:
            json.dump(doc2index, file)

    def save_keywords(self):
        """
        keywords处理，保存
        :return:
        """
        keywords = sorted(list(set(self.keywords)))
        keywords_trans = []
        for keyword in tqdm(keywords):
            if keyword.count(' ') <= 5:
                keywords_trans.append(keyword)
        # 去空
        keywords_trans = list(filter(not_empty, keywords_trans))
        print('num of keywords base:', len(keywords_trans))

        with open(self.keyword_path, 'w', encoding='UTF-8') as file:
            json.dump(keywords_trans, file)


if __name__ == '__main__':
    label = 'patent'
    doc_path = '../data/1.keyword_get/cnc_doc_' + label + '.txt'
    doc_index_path = '../data/1.keyword_get/cnc_doc_' + label + '2index.json'
    keyword_path = '../data/1.keyword_get/cnc_keywords_base_' + label + '.txt'

    data_process = DataProcess(label, doc_path, doc_index_path, keyword_path)
    data_process.get_date()
    data_process.save_keywords()
