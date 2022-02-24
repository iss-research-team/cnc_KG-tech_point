#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/30 下午4:52
# @Author  : liu yuhan
# @FileName: 3.combine.py
# @Software: PyCharm


import json
import numpy as np
import csv
import sys

from sklearn.cluster import DBSCAN
from collections import defaultdict

if __name__ == '__main__':
    label = sys.argv[1]
    node_feature_path = "../data/2.get_layer/node_emd_net_" + label + ".npy"
    x = np.load(node_feature_path, encoding="latin1")
    node_feature_path = "../data/input/cnc_keywords_" + label + ".json"
    with open(node_feature_path, 'r', encoding='UTF-8') as file:
        node_list = json.load(file)

    y_pred = DBSCAN(eps=10, min_samples=1).fit_predict(x)
    keyword_set = defaultdict(set)

    for index, cluster in enumerate(y_pred.tolist()):
        keyword_set[cluster].add(node_list[index])

    count = 0
    csv_write_path = "../data/input/keyword_combine_" + label + ".csv"
    csv_combine = csv.writer(open(csv_write_path, 'w', encoding='UTF-8', newline=''))
    for _, keyword in keyword_set.items():
        # print(keyword)
        if len(keyword) > 1:
            count += 1
            csv_combine.writerow(list(keyword))
    print('num of couple', count)
