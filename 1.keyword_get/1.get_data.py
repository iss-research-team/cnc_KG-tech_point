#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/8 下午4:46
# @Author  : liu yuhan
# @FileName: 1.get_data.py
# @Software: PyCharm

import json
import re
import spacy
from tqdm import tqdm

print("test")
print("刘铭到此一游")

def not_empty(s):
    return s and s.strip()


class DataProcess:
    '''
    类说明：
    对文本的进一步的预处理
    '''

    def __init__(self, label, doc_path, keyword_path):
        self.label = label
        print('label:',label)
        self.doc_path = doc_path
        self.keyword_path = keyword_path

        # 处理过程中用到的正则表达式子
        self.pattern_patent = re.compile(r'\(.*?\)')
        self.pattern_lite = re.compile(r'\[.*?]')
        self.pattern_sp = re.compile(r'\s\W+\s')

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
        '''
        Returns a list of stopwords.
        '''
        f = open("stopwords.txt")
        stopwords = set()
        for line in f:
            stopwords.add(line.rstrip())
        self.stopwords = stopwords

    def doc_deal(self, doc):
        """
        获取句子或者词语的词形还原结果
        :return:
        """
        doc = self.lemma_tool(doc)
        doc = ' '.join([token.lemma_ for token in doc])
        #  恢复连字符
        doc = doc.replace(' - ', '-')
        doc = ' '.join([word for word in doc.split() if word not in self.stopwords]) + ' '
        return doc

    def keyword_deal(self, doc):
        """
        获取句子或者词语的词形还原结果
        :return:
        """
        doc = re.sub(r"[,.;:!?|/\\]", ' ', doc)
        doc = self.lemma_tool(doc)
        doc = ' '.join([token.lemma_ for token in doc])
        #  恢复连字符
        doc = doc.replace(' - ', '-')
        doc = ' '.join([word for word in doc.split() if word not in self.stopwords])
        return doc

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
            for i, keyword in enumerate(keyword_temper):
                keyword_temper[i] = self.keyword_deal(keyword)
        return keyword_temper

    def get_keywords_lite(self, keyword_temper):
        """
        论文关键词的提取
        :param keywords_temper:
        :return:
        """
        keyword_temper = keyword_temper.split('; ')
        keyword_temper = [keyword.strip() for keyword in keyword_temper]

        return keyword_temper

    def doc_trans_patent(self, doc):
        """
        针对专利数据
        对单一句子的处理
        :param doc:
        :return:
        """
        doc = doc.strip()
        doc = re.sub(r"['\"]", ' ', doc)
        # 需要解决括号的问题
        char_list = re.findall(self.pattern_patent, doc)
        for char in char_list:
            # 规则可以继续叠加
            if char[1].isdigit() or \
                    'non-English language text' in char:
                        doc = doc.replace(char, '')
        doc = ' '.join(doc.split())
        doc = doc.replace('..', '.').replace('. .', '.')
        # 找到倒数两个句号，中间的话就是最后一句话
        doc_a, keywords_str = self.get_last_s(doc)
        # 修改这个判断的部分
        keywords_temper = self.get_keywords_patent(keywords_str)
        if keywords_temper:
            doc = self.doc_deal(doc_a)
        else:
            doc = self.doc_deal(doc)
        # 去除一些奇怪的连接符号
        doc = re.sub(self.pattern_sp, '. ', doc)
        return doc.strip(), keywords_temper

    def doc_trans_lite(self, doc):
        """
        论文的数据预处理相对简单
        :param doc:
        :return:
        """
        doc = doc.strip()
        doc = re.sub(r"['\"]", ' ', doc)
        # 需要解决括号的问题
        char_list = re.findall(self.pattern_lite, doc)
        for char in char_list:
            # 规则可以继续叠加
            if char[1].isdigit():
                doc = doc.replace(char, '')
        doc = ' '.join(doc.split())
        doc = doc.replace('..', '.').replace('. .', '.')

        doc = self.doc_deal(doc)
        doc = re.sub(self.pattern_sp, '. ', doc)
        return doc.strip()

    def get_date(self):
        """
        获取文本数据
        :return:
        """
        with open('../data/input/doc_dict_' + self.label + '.json', 'r', encoding='UTF-8') as file:
            data_dict = json.load(file)
        # write
        f_write = open(self.doc_path, 'w', encoding='UTF-8')
        print('文本数量：', len(data_dict))
        count = 0
        for t, inf in tqdm(data_dict.items()):
            # 分流，专利处理
            doc = ''
            keyword = []
            if self.label == 'patent':
                doc = inf['doc']
                # 关键词来自于文本的最后一句话
                doc, keyword = self.doc_trans_patent(doc)
            # 论文处理
            if self.label == 'literature':
                doc = t + ' ' + inf['doc']
                doc = self.doc_trans_lite(doc)
                keyword = inf['key_words_author']
                keyword = self.get_keywords_lite(keyword)

            f_write.write(doc + '\n')
            self.keywords += keyword
            # count += 1
            # if count == 500:
            #     break

        f_write.close()

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
    doc_path = '../data/input/cnc_doc_' + label + '.txt'
    keyword_path = '../data/input/cnc_keywords_base_' + label + '.txt'

    data_process = DataProcess(label, doc_path, keyword_path)
    data_process.get_date()
    data_process.save_keywords()

    # this is a test!!
    # 123
