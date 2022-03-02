#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/30 下午4:52
# @Author  : liu yuhan
# @FileName: 3.combine_check.py
# @Software: PyCharm


import json
import numpy as np
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import csv
import networkx as nx

from sklearn.cluster import DBSCAN
from collections import defaultdict
from tqdm import tqdm


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

    return len(index2embed)


def get_couple_inf(label):
    node_feature_path = '../data/2.layer_get/node_emb_net_' + label + '.npy'
    x = np.load(node_feature_path, encoding="latin1")
    node_path = '../data/1.keyword_get/cnc_keywords_' + label + '.json'
    with open(node_path, 'r', encoding='UTF-8') as file:
        node_list = json.load(file)

    # eps为距离阈值ϵ，min_samples为邻域样本数阈值MinPts,X为数据
    eps_list = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 40, 50]
    ms_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_cluster = np.zeros((len(eps_list), len(ms_list)))
    num_cluster_dict = dict()
    for i, eps in tqdm(enumerate(eps_list)):
        for j, ms in enumerate(ms_list):
            y_pred = DBSCAN(eps=eps, min_samples=ms).fit_predict(x)

            keyword_set = defaultdict(set)
            for index, cluster in enumerate(y_pred.tolist()):
                keyword_set[cluster].add(node_list[index])

            count = 0
            for _, keyword in keyword_set.items():
                if len(keyword) > 1:
                    count += 1

            num_cluster[i][j] = count
            num_cluster_dict['eps_' + str(eps) + '_ms_' + str(ms)] = count
            # print('eps', eps, 'ms', ms, 'num of cluster', len(keyword_set), 'num of couple', count)

    df = pd.DataFrame(num_cluster, index=eps_list, columns=ms_list)
    ax = sns.heatmap(df, annot=True)
    plt.savefig('../data/fig/heatmap_' + label + '.png')

    return sorted(num_cluster_dict.items(), key=lambda x: x[1], reverse=True)


def make_coule_file(label, eps, ms):
    node_feature_path = "../data/2.layer_get/node_emb_net_" + label + ".npy"
    x = np.load(node_feature_path, encoding="latin1")
    node_feature_path = "../data/1.keyword_get/cnc_keywords_" + label + ".json"
    with open(node_feature_path, 'r', encoding='UTF-8') as file:
        node_list = json.load(file)

    y_pred = DBSCAN(eps=eps, min_samples=ms).fit_predict(x)
    keyword_set = defaultdict(set)

    for index, cluster in enumerate(y_pred.tolist()):
        keyword_set[cluster].add(node_list[index])

    keyword_set_list = list(keyword_set.values())
    keyword_set_list = [list(keywords) for keywords in keyword_set_list]

    return keyword_set_list


def get_parameter(par_str):
    par_str = par_str.split("_")
    return float(par_str[1]), int(par_str[3])


if __name__ == '__main__':
    couple_inf_dict_pat = get_couple_inf('patent')
    couple_inf_dict_lit = get_couple_inf('literature')
    # print(couple_inf_dict_pat)
    # print(couple_inf_dict_lit)

    index_len_map = np.zeros((len(couple_inf_dict_pat), len(couple_inf_dict_lit)))

    for i, para_p in tqdm(enumerate(couple_inf_dict_pat)):
        for j, para_l in enumerate(couple_inf_dict_lit):
            esp_p, ms_p = get_parameter(para_p[0])
            esp_l, ms_l = get_parameter(para_l[0])
            combine_list_p = make_coule_file('patent', esp_p, ms_p)
            combine_list_l = make_coule_file('literature', esp_l, ms_l)
            combine_list = combine_combine_list(combine_list_p, combine_list_l)

            # 最终的获取index
            keyword2index_path = "../data/2.layer_get/keyword2index.json"
            index2embed_path = "../data/2.layer_get/index2embed.json"

            index_len = get_index(combine_list, keyword2index_path, index2embed_path)
            index_len_map[i][j] = index_len

    couple_inf_dict_pat = [para[0] for para in couple_inf_dict_pat]
    couple_inf_dict_lit = [para[0] for para in couple_inf_dict_lit]

    df = pd.DataFrame(index_len_map, index=couple_inf_dict_pat, columns=couple_inf_dict_lit)
    ax = sns.heatmap(df, annot=True)
    plt.savefig('../data/fig/heatmap_index_len.png')
