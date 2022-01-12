#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/16 上午5:37
# @Author  : liu yuhan
# @FileName: 3.dis.py
# @Software: PyCharm


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import json
import networkx as nx
import torch
import torch.nn.functional as F
from tqdm import tqdm


def get_cos_similar(v1: list, v2: list):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


def get_dis(k):
    '''
    逻辑和之前的写法类似，构建一个网络，求联通体
    :return:
    '''
    # 收集连接
    link_list = []
    # 收集dis，观察分布
    dis_list = []
    node_emb = np.load('../data/output/node_emd_2.npy')
    node_size = node_emb.shape[0]
    for i in tqdm(range(0, node_size - 1)):
        for j in range(i + 1, node_size):
            dis = F.cosine_similarity(torch.FloatTensor(node_emb[i]), torch.FloatTensor(node_emb[j]), dim=0)
            dis_list.append(dis)
            if dis > k:
                link_list.append([i, j])
    print(k, len(link_list))

    return dis_list, link_list


def dis_draw(dis_list):
    plt.style.use('ggplot')

    sns.distplot(dis_list, hist=False, kde=False, fit=stats.norm,
                 fit_kws={'color': 'black', 'label': 'dis', 'linestyle': '-'})

    # 呈现图例
    plt.legend()
    # 呈现图形
    plt.show()


def combine(link_list, node_label_dict):
    '''
    通过最大联通图的方式对
    :param couple_list:
    :param node_label_dict:
    :return:
    '''
    link_list_trans = []
    for link in link_list:
        link_list_trans.append([node_label_dict[link[0]], node_label_dict[link[1]]])

    G = nx.Graph()
    G.add_edges_from(link_list_trans)
    nx.connected_components(G)

    combine_list = []

    for c in nx.connected_components(G):
        g = G.subgraph(c)
        combine_list.append(list(g.nodes()))

    print(combine_list)
    print(len(combine_list))

    with open('../data/output/combine_list.json', 'w', encoding='UTF-8') as file:
        json.dump(combine_list, file)


if __name__ == '__main__':
    k = 0.5
    dis_list, link_list = get_dis(k)
    dis_draw(dis_list)

    with open('../data/input/demo/keywords_patent_500.json', 'r', encoding='UTF-8') as file:
        node_list = json.load(file)
    print(len(node_list))
    node_dict = dict(zip([i for i in range(len(node_list))], node_list))
    combine(link_list, node_dict)
