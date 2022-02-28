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


def get_index(combine_list, keyword2index_path, index2embed_path):
    # 构建两个字典
    # 一个用于存储embed，一个存储词表
    keyword2index = dict()
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
            if node in keyword_dict_patent:
                emb += np.array(keyword_dict_patent[node])
            if node in keyword_dict_literature:
                emb += np.array(keyword_dict_literature[node])
        index2embed[index] = emb.tolist()
        index += 1

    for node in keyword_list:
        if node in keyword2index:
            continue
        emb = np.zeros(512)
        if node in keyword_dict_patent:
            emb += np.array(keyword_dict_patent[node])
        if node in keyword_dict_literature:
            emb += np.array(keyword_dict_literature[node])
        index2embed[index] = emb.tolist()
        index += 1

    print('关键词数量：', len(index2embed))

    with open(keyword2index_path, 'w', encoding='UTF-8') as file:
        json.dump(keyword2index, file)
    with open(index2embed_path, 'w', encoding='UTF-8') as file:
        json.dump(index2embed, file)


if __name__ == '__main__':
    """
    生成两个字典：
    1.index-keyword
    2.index-vector
    """
    # push test
    combine_path_p = "../data/2.layer_get/cnc_keyword_combine_patent.csv"
    combine_path_l = "../data/2.layer_get/cnc_keyword_combine_literature.csv"
    combine_list_p = get_combine_list(combine_path_p)
    combine_list_l = get_combine_list(combine_path_l)
    combine_list = combine_combine_list(combine_list_p, combine_list_l)

    # 最终的获取index
    keyword2index_path = "../data/2.layer_get/keyword2index.json"
    index2embed_path = "../data/2.layer_get/index2embed.json"

    get_index(combine_list, keyword2index_path, index2embed_path)
