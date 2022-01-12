#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/12 上午11:13
# @Author  : liu yuhan
# @FileName: 4.get_index.py
# @Software: PyCharm

import csv
from collections import defaultdict


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
    # 构建一个字典
    kw2index = dict()
    for index, kw_list in enumerate(combine_list_1):
        for kw in kw_list:
            kw2index[kw] = index
    # 将第二个字典加入第一个字典
    length = len(combine_list_1)
    for kw_list in combine_list_2:
        for kw in kw_list:
            if kw in kw2index:
                # 在已有的字典中已经出现
                index = kw2index[kw]
                break
        else:
            # 在字典中未出现，index2kw长度加1
            index = length
            length += 1

        for kw in kw_list:
            kw2index[kw] = index

    index2kw = defaultdict(set)
    for kw, index in kw2index.items():
        index2kw[index].add(kw)

    return index2kw

    # def get_index2kw(kw_list, combine_list):


if __name__ == '__main__':
    """
    生成两个字典：
    1.index-keyword
    2.index-vector
    """
    combine_path_p = "../data/input/keyword_combine_patent.csv"
    combine_path_l = "../data/input/keyword_combine_literature.csv"

    combine_list_p = get_combine_list(combine_path_p)
    combine_list_l = get_combine_list(combine_path_l)
    print(len(combine_list_p))
    print(len(combine_list_l))

    index2kw = combine_combine_list(combine_list_p, combine_list_l)

    print(len(index2kw))
    print(index2kw)
