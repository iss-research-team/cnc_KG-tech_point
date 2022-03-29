#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/30 下午4:52
# @Author  : liu yuhan
# @FileName: 3.combine_check.py
# @Software: PyCharm


import json
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from collections import defaultdict
from tqdm import tqdm


class Index:
    def __init__(self, if_trans):
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
        self.keyword2embed = dict()
        for keyword in self.keyword_list:
            emb = np.zeros(512)
            if keyword in self.keyword_dict_patent:
                emb += np.array(self.keyword_dict_patent[keyword])
            if keyword in self.keyword_dict_literature:
                emb += np.array(self.keyword_dict_literature[keyword])
            self.keyword2embed[keyword] = emb

        self.combine_list = []
        # index
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
        y_pred = DBSCAN(eps=eps, min_samples=ms).fit_predict(np.array(list(self.keyword2embed.values())))
        keyword_set = defaultdict(set)

        for index, cluster in enumerate(y_pred.tolist()):
            if cluster == -1:
                continue
            keyword_set[cluster].add(self.keyword_list[index])
        keyword_set_list = list(keyword_set.values())
        self.combine_list = [list(keywords) for keywords in keyword_set_list]

        count = 0
        for combine in self.combine_list:
            if len(combine) > 1:
                count += 1

        return count


if __name__ == '__main__':

    # 最终的获取index
    if_trans = 'yes'
    index = Index(if_trans)
    # eps为距离阈值ϵ，min_samples为邻域样本数阈值MinPts,X为数据
    eps_list = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 40, 50]
    ms_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_cluster = np.zeros((len(eps_list), len(ms_list)))
    num_cluster_dict = dict()

    for i, eps in tqdm(enumerate(eps_list)):
        for j, ms in enumerate(ms_list):
            num_cluster[i][j] = index.get_combine_list(eps=eps, ms=ms)

    df = pd.DataFrame(num_cluster, index=eps_list, columns=ms_list)
    if if_trans == 'yes':
        save_csv = '../fig/2.layer_get/heatmap_trans.csv'
        save_fig = '../fig/2.layer_get/heatmap_trans.png'
    else:
        save_csv = '../fig/2.layer_get/heatmap_origin.csv'
        save_fig = '../fig/2.layer_get/heatmap_origin.png'
    df.to_csv(save_csv)
    ax = sns.heatmap(df, annot=True)
    plt.show()
    plt.savefig(save_fig)
