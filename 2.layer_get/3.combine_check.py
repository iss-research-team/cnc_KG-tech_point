#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/30 下午4:52
# @Author  : liu yuhan
# @FileName: 3.combine_check.py
# @Software: PyCharm


import json
import numpy as np
import csv
from sklearn.cluster import DBSCAN
from collections import defaultdict
from tqdm import tqdm

if __name__ == '__main__':
    label = 'literature'
    node_feature_path = '../data/2.get_layer/node_emd_net_' + label + '.npy'
    x = np.load(node_feature_path, encoding="latin1")
    node_feature_path = '../data/input/cnc_keywords_' + label + '.json'
    with open(node_feature_path, 'r', encoding='UTF-8') as file:
        node_list = json.load(file)

    # eps为距离阈值ϵ，min_samples为邻域样本数阈值MinPts,X为数据
    eps_list = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 40, 50]
    ms_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    num_cluster = np.zeros((len(eps_list), len(ms_list)))
    for eps in tqdm(eps_list):
        for ms in ms_list:
            y_pred = DBSCAN(eps=eps, min_samples=ms).fit_predict(x)

            keyword_set = defaultdict(set)
            for index, cluster in enumerate(y_pred.tolist()):
                keyword_set[cluster].add(node_list[index])
            count = 0
            for _, keyword in keyword_set.items():
                if len(keyword) > 1:
                    count += 1
            print('eps', eps, 'ms', ms, 'num of cluster', len(keyword_set), 'num of couple', count)
