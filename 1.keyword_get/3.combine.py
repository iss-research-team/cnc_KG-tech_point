#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/8 上午11:19
# @Author  : liu yuhan
# @FileName: 3.combine_check.py
# @Software: PyCharm

import json


def get_keywords(path):
    '''
    Returns a list of stopwords.
    '''
    f = open(path)
    stopwords = []
    for line in f:
        stopwords.append(line.strip())
    return stopwords


if __name__ == '__main__':
    input_path_p = '../data/1.keyword_get/cnc_keywords_multiple_patent.txt'
    input_path_l = '../data/1.keyword_get/cnc_keywords_multiple_literature.txt'

    save_path_p = '../data/1.keyword_get/cnc_keywords_patent.json'
    save_path_l = '../data/1.keyword_get/cnc_keywords_literature.json'

    node_p = get_keywords(input_path_p)
    node_l = get_keywords(input_path_l)
    node_list = sorted(list(set(node_p) | set(node_l)))
    # save
    with open(save_path_p, 'w', encoding='UTF-8') as file:
        json.dump(node_p, file)
    with open(save_path_l, 'w', encoding='UTF-8') as file:
        json.dump(node_l, file)
