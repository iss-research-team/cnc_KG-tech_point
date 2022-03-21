#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/12 上午11:13
# @Author  : liu yuhan
# @FileName: 4.get_index.py
# @Software: PyCharm

import csv
import json
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from collections import defaultdict


def make_coule_file(label, eps, ms):
    node_feature_path = "../data/2.layer_get/node_emb_net_" + label + ".npy"
    x = np.load(node_feature_path, encoding="latin1")
    node_feature_path = "../data/1.keyword_get/cnc_keywords_" + label + ".json"
    with open(node_feature_path, 'r', encoding='UTF-8') as file:
        node_list = json.load(file)

    y_pred = DBSCAN(eps=eps, min_samples=ms).fit_predict(x)
    keyword_set = defaultdict(set)

    for index, cluster in enumerate(y_pred.tolist()):
        if cluster == -1:
            continue
        keyword_set[cluster].add(node_list[index])

    keyword_set_list = list(keyword_set.values())
    keyword_set_list = [list(keywords) for keywords in keyword_set_list]

    return keyword_set_list


def get_combine_list(combine_path):
    """
    将csv转换成list
    :param combine_path:
    :return:
    """
    combine_list = []
    csv_reader = csv.reader(open(combine_path, "r", encoding="UTF-8"))
    for line in csv_reader:
        combine_list.append(line)
    return combine_list


def combine_combine_list(combine_list_1, combine_list_2):
    """
    组合两个list
    :param combine_list_1:
    :param combine_list_2:
    :return:
    """
    g = nx.Graph()
    for combine in combine_list_1 + combine_list_2:
        g.add_nodes_from(combine)
        for i in range(0, len(combine) - 1):
            for j in range(i + 1, len(combine)):
                g.add_edge(combine[i], combine[j])

    combine_list = []
    for g_sub in nx.connected_components(g):
        # 得到不连通的子集
        combine_list.append(g.subgraph(g_sub).nodes())

    return combine_list


def get_index(combine_list, keyword2index_path, index2keyword_path, index2embed_path):
    # 构建两个字典
    # 一个用于存储embed，一个存储词表
    keyword2index = dict()
    index2keyword = defaultdict(set)
    index2embed = dict()

    with open('../data/1.keyword_get/cnc_keywords_patent.json', 'r', encoding='UTF-8') as file:
        keyword_list_patent = json.load(file)
    with open('../data/1.keyword_get/cnc_keywords_literature.json', 'r', encoding='UTF-8') as file:
        keyword_list_literature = json.load(file)

    keyword_embed_patent_path = "../data/2.layer_get/node_emb_net_patent.npy"
    keyword_embed_patent = np.load(keyword_embed_patent_path, encoding="latin1").tolist()
    keyword_embed_literature_path = "../data/2.layer_get/node_emb_net_literature.npy"
    keyword_embed_literature = np.load(keyword_embed_literature_path, encoding="latin1").tolist()

    keyword_dict_patent = dict(zip(keyword_list_patent, keyword_embed_patent))
    keyword_dict_literature = dict(zip(keyword_list_literature, keyword_embed_literature))

    keyword_list = sorted(list(set(keyword_list_patent + keyword_list_literature)))

    # 先对combine_list进行处理
    index = 0
    for combine in combine_list:
        emb = np.zeros(512)
        for node in combine:
            keyword2index[node] = index
            index2keyword[index].add(node)
            if node in keyword_dict_patent:
                emb += np.array(keyword_dict_patent[node])
            if node in keyword_dict_literature:
                emb += np.array(keyword_dict_literature[node])
        index2embed[index] = emb.tolist()
        index += 1

    # DBS聚类过程中可能会出现
    for node in keyword_list:
        if node in keyword2index:
            continue
        keyword2index[node] = index
        index2keyword[index].add(node)
        emb = np.zeros(512)
        if node in keyword_dict_patent:
            emb += np.array(keyword_dict_patent[node])
        if node in keyword_dict_literature:
            emb += np.array(keyword_dict_literature[node])
        index2embed[index] = emb.tolist()
        index += 1

    print('关键词数量：', len(index2embed))
    index2keyword = dict((index, list(keyword_set)) for index, keyword_set in index2keyword.items())

    with open(keyword2index_path, 'w', encoding='UTF-8') as file:
        json.dump(keyword2index, file)
    with open(index2keyword_path, 'w', encoding='UTF-8') as file:
        json.dump(index2keyword, file)
    with open(index2embed_path, 'w', encoding='UTF-8') as file:
        json.dump(index2embed, file)


if __name__ == '__main__':
    """
    生成两个字典：
    1.index-keyword
    2.index-vector
    """
    esp_p, ms_p = 10, 1
    esp_l, ms_l = 10, 1
    combine_list_p = make_coule_file('patent', esp_p, ms_p)
    combine_list_l = make_coule_file('literature', esp_l, ms_l)
    combine_list = combine_combine_list(combine_list_p, combine_list_l)

    # 最终的获取index
    keyword2index_path = "../data/2.layer_get/keyword2index.json"
    index2keyword_path = "../data/2.layer_get/index2keyword.json"
    index2embed_path = "../data/2.layer_get/index2embed.json"

    get_index(combine_list, keyword2index_path, index2keyword_path, index2embed_path)
