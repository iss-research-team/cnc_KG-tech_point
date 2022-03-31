#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/2/25 下午4:49
# @Author  : liu yuhan
# @FileName: 5.get_layer.py
# @Software: PyCharm

import json
import numpy as np
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import networkx as nx


def set2list(inf_dict):
    return dict((key, list(value_set)) for key, value_set in inf_dict.items())


def clustering():
    """
    生成技术树
    :param data:
    :return:
    """
    # 数据获取
    index2embed_path = "../data/2.layer_get/index2embed_trans.json"
    with open(index2embed_path, 'r', encoding='UTF-8') as file:
        index2embed = json.load(file)
    data = np.zeros((len(index2embed), 512))
    for index, embed in index2embed.items():
        data[int(index)] = embed
    # 层次聚类
    Z = linkage(data, method='ward', metric='euclidean')
    # 存储
    keyword2index_path = "../data/2.layer_get/keyword2index_trans.json"
    with open(keyword2index_path, 'r', encoding='UTF-8') as file:
        keyword2index = json.load(file)

    tech_tree_1 = dict()
    tech_tree_2 = defaultdict(set)
    tech_class_set = set()

    tech_node_index = len(index2embed)

    # 1级技术获取
    f = fcluster(Z, 20, 'distance')
    print('num of class 1', np.max(f))
    for index, c in enumerate(f):
        tech_node = 'class_1_{}'.format(c)
        if tech_node not in keyword2index:
            keyword2index[tech_node] = tech_node_index
            tech_node_index += 1
            tech_class_set.add(tech_node)
        tech_tree_1[index] = keyword2index[tech_node]
        tech_tree_2[keyword2index[tech_node]].add(index)

    # 2级技术获取
    f = fcluster(Z, 229, 'distance')
    print(np.max(f))
    for index, c in enumerate(f):
        tech_node = 'class_2_{}'.format(c)
        if tech_node not in keyword2index:
            keyword2index[tech_node] = tech_node_index
            tech_node_index += 1
            tech_class_set.add(tech_node)
        tech_tree_1[tech_tree_1[index]] = keyword2index[tech_node]
        tech_tree_2[keyword2index[tech_node]].add(tech_tree_1[index])

    # 3级技术获取
    f = fcluster(Z, 1210, 'distance')
    print(np.max(f))
    for index, c in enumerate(f):
        tech_node = 'class_3_{}'.format(c)
        if tech_node not in keyword2index:
            keyword2index[tech_node] = tech_node_index
            tech_node_index += 1
            tech_class_set.add(tech_node)
        tech_tree_1[tech_tree_1[tech_tree_1[index]]] = keyword2index[tech_node]
        tech_tree_2[keyword2index[tech_node]].add(tech_tree_1[tech_tree_1[index]])

    # 3级技术获取
    f = fcluster(Z, 10000, 'distance')
    for index, c in enumerate(f):
        tech_node = 'class_4_{}'.format(c)
        if tech_node not in keyword2index:
            keyword2index[tech_node] = tech_node_index
            tech_node_index += 1
            tech_class_set.add(tech_node)
        tech_tree_1[tech_tree_1[tech_tree_1[tech_tree_1[index]]]] = keyword2index[tech_node]
        tech_tree_2[keyword2index[tech_node]].add(tech_tree_1[tech_tree_1[tech_tree_1[index]]])

    tech_tree_2 = set2list(tech_tree_2)

    return tech_tree_1, tech_tree_2, keyword2index, list(tech_class_set)


def tech_tree_draw(tech_tree):
    node_set = set()
    link_list = []

    for node_1, node_2 in tech_tree.items():
        node_set.add(node_1)
        node_set.add(node_2)
        link_list.append([node_1, node_2])

    node_list = list(node_set)
    print(len(node_list))
    print(len(link_list))

    g_tech = nx.Graph()
    g_tech.add_nodes_from(node_list)
    g_tech.add_edges_from(link_list)

    pos = nx.spring_layout(g_tech)
    nx.draw(g_tech, pos=pos, node_size=2)
    plt.show()


def train_data_make(tech_tree, keyword2index, tech_class_set):
    """
    生成一个可以用于联合抽取的样本
    :param tech_tree: 
    :return: 
    """
    index2keyword = defaultdict(set)
    for keyword, index in keyword2index.items():
        index2keyword[index].add(keyword)

    index2keyword = set2list(index2keyword)
    train_data_list = []

    for node_f, node_s_list in tech_tree.items():
        if index2keyword[node_s_list[0]][0] in tech_class_set:
            continue
        node_s_list_trans = []
        for node_s in node_s_list:
            node_s_list_trans += index2keyword[node_s]
        num_node_s = len(node_s_list_trans)
        for i in range(0, num_node_s - 1):
            for j in range(i + 1, num_node_s):
                train_data_list.append([node_s_list_trans[i], node_s_list_trans[j]])
    print(train_data_list)


if __name__ == '__main__':
    tech_tree_1, tech_tree_2, keyword2index, tech_class_set = clustering()
    # print('drawing---')
    # tech_tree_draw(tech_tree_2, keyword2index, tech_class_set)
    train_data_make(tech_tree_2, keyword2index, tech_class_set)
