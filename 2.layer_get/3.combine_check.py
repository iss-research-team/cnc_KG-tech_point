#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/30 下午4:52
# @Author  : liu yuhan
# @FileName: 3.combine_check.py
# @Software: PyCharm


import json
import numpy as np
import sys
from sklearn.cluster import DBSCAN
from collections import defaultdict
from tqdm import tqdm

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    label = sys.argv[1]
    node_feature_path = '../data/2.layer_get/node_emb_net_' + label + '.npy'
    x = np.load(node_feature_path, encoding="latin1")
    node_path = '../data/1.keyword_get/cnc_keywords_' + label + '.json'
    with open(node_path, 'r', encoding='UTF-8') as file:
        node_list = json.load(file)

    # eps为距离阈值ϵ，min_samples为邻域样本数阈值MinPts,X为数据
    eps_list = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 40, 50]
    ms_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_cluster = np.zeros((len(eps_list), len(ms_list)))
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

            # print('eps', eps, 'ms', ms, 'num of cluster', len(keyword_set), 'num of couple', count)

    df = pd.DataFrame(num_cluster, index=eps_list, columns=ms_list)
    ax = sns.heatmap(df)
    plt.savefig('../data/fig/heatmap_' + label + '.png')
