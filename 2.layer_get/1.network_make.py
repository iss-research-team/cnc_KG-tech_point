#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 下午4:42
# @Author  : liu yuhan
# @FileName: 1.network_make-v2.py
# @Software: PyCharm


# 后续的修改计划
# 1.几个关键参数需要放在外面，

import json
import re
import scipy.stats
import numpy as np
import sys

from collections import Counter
from collections import defaultdict
from tqdm import tqdm


def dis_flag(dis, dis_ave):
    flag = 0
    # print(dis)
    if 0 < dis < dis_ave:
        flag = 1
    return flag


class NetworkMaker:
    def __init__(self, label, if_trans, keyword_path, doc_path, save_path):
        self.label = label
        self.if_trans = if_trans
        with open(keyword_path, 'r', encoding='UTF-8') as file:
            node_list = json.load(file)
        self.node_list = [' ' + node + ' ' for node in node_list]
        self.node_num = len(self.node_list)
        print('节点数量：', self.node_num)
        self.node_dict = dict(zip(self.node_list, [i for i in range(len(node_list))]))
        self.doc_path = doc_path
        self.save_path = save_path
        if if_trans == 'yes':
            # 下位邻居
            self.node_neighbor_dict_1 = defaultdict(list)
            # 上位邻居
            self.node_neighbor_dict_2 = defaultdict(list)
        self.link_list_counter = Counter()
        self.link_list_weighted = []

    def get_keyword_seq(self, s):
        """
        通过每一句话获取节点的邻居
        :param s:
        :return:
        """
        node_list = []
        node_start_list = []
        node_length_list = []
        for node in self.node_dict:
            if node in s:
                node_length = node.count(' ') - 1
                bit = 0
                for i in range(s.count(node)):
                    start_bit = s.find(node, bit)
                    node_list.append(self.node_dict[node])
                    node_start_list.append(s[:start_bit].count(' '))
                    node_length_list.append(node_length)
                    bit = start_bit + 1
            else:
                continue
        # 获取邻居
        node_num = len(node_list)
        s_l = s.count(' ') - 1

        # 关键参数1
        if node_num:
            dis_threshold = s_l / node_num
            # print('k', dis_threshold)
            # dis_threshold = 100000
        else:
            dis_threshold = 0

        for i in range(node_num):
            for j in range(node_num):
                if i == j:
                    # 1.避开同一个词
                    continue
                if node_list[i] == node_list[j]:
                    # 2.重复词语的问题
                    continue
                # ——————————————————————————————————————————————————————————
                # 这里更好的设计是通过关键词之间的距离设计一种权重：
                # A....B
                # C..B
                # 在构建A和C之间的连接的时候，考虑ab之间的距离和cb之间的距离不同
                # ——————————————————————————————————————————————————————————
                # 计算距离
                # 2021/12/21
                # 通过平均距离来限定还是太小，
                # 目前有两个方案
                # 1.通过窗口，设置一个窗口大小，窗口之外的词语不予以考虑
                # 2.进行更小范围的语言切片
                # 方案2先不予以考虑.

                if node_start_list[i] < node_start_list[j]:
                    dis = node_start_list[j] - node_start_list[i] - node_length_list[i]
                    if dis_flag(dis, dis_threshold):
                        self.node_neighbor_dict_1[node_list[i]].append(node_list[j])
                else:
                    dis = node_start_list[i] - node_start_list[j] - node_length_list[j]
                    if dis_flag(dis, dis_threshold):
                        self.node_neighbor_dict_2[node_list[i]].append(node_list[j])

    def get_link(self, i, j):
        """
        这是一个很简单的版本
        :param i:
        :param j:
        :return:
        """
        node_i_1 = self.node_neighbor_dict_1[i]
        node_i_2 = self.node_neighbor_dict_2[i]
        node_j_1 = self.node_neighbor_dict_1[j]
        node_j_2 = self.node_neighbor_dict_2[j]

        weight = len(set(node_i_1) & set(node_j_1)) + len(set(node_i_2) & set(node_j_2))
        return weight

    def get_keyword_seq_simple(self, s):
        """
        if_trans == no
        :param s:
        :return:
        """
        node_list = []
        for node in self.node_dict:
            if node not in s:
                continue
            for i in range(s.count(node)):
                node_list.append(self.node_dict[node])
        return sorted(node_list)

    def link_trans(self):
        """
        add norm
        """
        if self.label == "patent":
            weight_set = 600
        else:
            weight_set = 150
        print("weight:", weight_set)
        for link, weight in self.link_list_counter.items():
            s, t = [int(node) for node in link.split(' | ')]
            if weight >= weight_set:  # 关键参数2
                self.link_list_weighted.append([s, t, weight])

    def network_make(self):
        """
        网络构建
        两个循环，
        一个循环用于获取邻居
        另一个循环用于网络构建
        :return:
        """
        doc_file = open(self.doc_path, 'r', encoding='UTF-8')

        if self.if_trans == 'yes':
            print('trans---')
            for sentence in tqdm(doc_file):
                self.get_keyword_seq(' ' + sentence.replace('.', '').strip() + ' ')
            # 连接构建
            for i in tqdm(range(0, self.node_num - 1)):
                for j in range(i + 1, self.node_num):
                    weight = self.get_link(i, j)
                    if weight:
                        self.link_list_counter[str(i) + ' | ' + str(j)] += weight
        else:
            print('origin---')
            for sentence in tqdm(doc_file):
                keyword_list = self.get_keyword_seq_simple(' ' + sentence.replace('.', '').strip() + ' ')
                if len(keyword_list) < 2:
                    continue
                num_keyword = len(keyword_list)
                for i in range(0, num_keyword - 1):
                    for j in range(i + 1, num_keyword):
                        if keyword_list[i] == keyword_list[j]:
                            continue
                        self.link_list_counter[str(keyword_list[i]) + ' | ' + str(keyword_list[j])] += 1

        # 字典转list
        self.link_trans()
        print('num of link:', len(self.link_list_weighted))
        with open(self.save_path, 'w', encoding='UTF-8') as file:
            json.dump(self.link_list_weighted, file)


if __name__ == '__main__':
    label = sys.argv[1]
    if_trans = sys.argv[2]
    keyword_path = '../data/1.keyword_get/keywords/cnc_keywords_' + label + '.json'
    doc_path = '../data/1.keyword_get/doc/cnc_doc_' + label + '.txt'
    if if_trans == 'yes':
        link_save_path = '../data/2.layer_get/link/cnc_keywords_link_' + label + '_trans.json'
    else:
        link_save_path = '../data/2.layer_get/link/cnc_keywords_link_' + label + '_origin.json'

    network_maker = NetworkMaker(label, if_trans, keyword_path, doc_path, link_save_path)
    network_maker.network_make()
