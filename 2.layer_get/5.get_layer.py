#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/2/25 下午4:49
# @Author  : liu yuhan
# @FileName: 5.get_layer.py
# @Software: PyCharm

import json
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


def get_data():
    index2embed_path = "../data/2.layer_get/index2embed.json"
    with open(index2embed_path, 'r', encoding='UTF-8') as file:
        index2embed = json.load(file)

    data = np.zeros((len(index2embed), 512))
    for index, embed in index2embed.items():
        data[int(index)] = embed

    return data


def clustering(data):
    plt.figure(figsize=(20, 6))
    Z = linkage(data, method='ward', metric='euclidean')
    p = dendrogram(Z, 0)
    plt.show()


if __name__ == '__main__':
    data = get_data()
    clustering(data)
