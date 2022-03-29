#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/12 上午11:13
# @Author  : liu yuhan
# @FileName: 4.get_index.py
# @Software: PyCharm

import json
import csv
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict


class Index:
    def __init__(self, if_trans, keyword2index_path, index2keyword_path, index2embed_path, combine_list_path):

        self.keyword2index_path = keyword2index_path
        self.index2keyword_path = index2keyword_path
        self.index2embed_path = index2embed_path
        self.combine_list_path = combine_list_path
        # node_list
        with open('../data/1.keyword_get/keywords/cnc_keywords_patent.json', 'r', encoding='UTF-8') as file:
            keyword_list_patent = json.load(file)
        with open('../data/1.keyword_get/keywords/cnc_keywords_literature.json', 'r', encoding='UTF-8') as file:
            keyword_list_literature = json.load(file)
        self.keyword_list = sorted(list(set(keyword_list_patent + keyword_list_literature)))
        # node_embed
        if if_trans == 'yes':
            keyword_embed_patent_path = "../data/2.layer_get/node_emb/node_emb_net_patent_trans.npy"
            keyword_embed_literature_path = "../data/2.layer_get/node_emb/node_emb_net_literature_trans.npy"
        else:
            keyword_embed_patent_path = "../data/2.layer_get/node_emb/node_emb_net_patent_origin.npy"
            keyword_embed_literature_path = "../data/2.layer_get/node_emb/node_emb_net_literature_origin.npy"

        keyword_embed_patent = np.load(keyword_embed_patent_path, encoding="latin1").tolist()
        keyword_embed_literature = np.load(keyword_embed_literature_path, encoding="latin1").tolist()

        self.keyword_dict_patent = dict(zip(keyword_list_patent, keyword_embed_patent))
        self.keyword_dict_literature = dict(zip(keyword_list_literature, keyword_embed_literature))
        self.combine_list = []
        # index
        self.keyword2embed = dict()
        self.keyword2index = dict()
        self.index2keyword = defaultdict(set)
        self.index2embed = dict()

    def get_combine_list(self, eps, ms):
        """
        通过聚类合并
        :param eps:
        :param ms:
        :return:
        """
        for keyword in self.keyword_list:
            emb = np.zeros(512)
            if keyword in self.keyword_dict_patent:
                emb += np.array(self.keyword_dict_patent[keyword])
            if keyword in self.keyword_dict_literature:
                emb += np.array(self.keyword_dict_literature[keyword])
            self.keyword2embed[keyword] = emb

        y_pred = DBSCAN(eps=eps, min_samples=ms).fit_predict(np.array(list(self.keyword2embed.values())))
        keyword_set = defaultdict(set)

        for index, cluster in enumerate(y_pred.tolist()):
            if cluster == -1:
                continue
            keyword_set[cluster].add(self.keyword_list[index])
        keyword_set_list = list(keyword_set.values())
        self.combine_list = [list(keywords) for keywords in keyword_set_list]

        csv_write = csv.writer(open(self.combine_list_path, 'w', encoding='UTF-8', newline=''))
        for combine in self.combine_list:
            if len(combine) > 1:
                csv_write.writerow(combine)

    def get_index(self, eps, ms):
        self.get_combine_list(eps, ms)
        # 先对combine_list进行处理
        index = 0
        for combine in self.combine_list:
            emb = np.zeros(512)
            for node in combine:
                self.keyword2index[node] = index
                self.index2keyword[index].add(node)
                emb += self.keyword2embed[node]
            self.index2embed[index] = emb.tolist()
            index += 1

        # DBS聚类过程中可能会出现-1
        for node in self.keyword_list:
            if node in self.keyword2index:
                continue
            self.keyword2index[node] = index
            self.index2keyword[index].add(node)
            self.index2embed[index] = self.keyword2embed[node].tolist()
            index += 1

        print('关键词数量：', len(self.index2embed))

        def set2list(inf_dict):
            return dict((key, list(value_set)) for key, value_set in inf_dict.items())

        self.index2keyword = set2list(self.index2keyword)

        with open(self.keyword2index_path, 'w', encoding='UTF-8') as file:
            json.dump(self.keyword2index, file)
        with open(self.index2keyword_path, 'w', encoding='UTF-8') as file:
            json.dump(self.index2keyword, file)
        with open(self.index2embed_path, 'w', encoding='UTF-8') as file:
            json.dump(self.index2embed, file)


if __name__ == '__main__':
    """
    生成两个字典：
    1.index-keyword
    2.index-vector
    """
    # 最终的获取index
    if_trans = 'yes'
    if if_trans == 'yes':
        keyword2index_path = "../data/2.layer_get/keyword2index_trans.json"
        index2keyword_path = "../data/2.layer_get/index2keyword_trans.json"
        index2embed_path = "../data/2.layer_get/index2embed_trans.json"
        combine_list_path = "../data/2.layer_get/combine_list_trans.csv"
    else:
        keyword2index_path = "../data/2.layer_get/keyword2index_origin.json"
        index2keyword_path = "../data/2.layer_get/index2keyword_origin.json"
        index2embed_path = "../data/2.layer_get/index2embed_origin.json"
        combine_list_path = "../data/2.layer_get/combine_list_origin.csv"

    index = Index(if_trans, keyword2index_path, index2keyword_path, index2embed_path, combine_list_path)
    index.get_index(eps=9, ms=1)
