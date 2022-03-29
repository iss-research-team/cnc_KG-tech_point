#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/8 上午11:19
# @Author  : liu yuhan
# @FileName: 3.combine_check.py
# @Software: PyCharm

import os
import pandas as pd


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
    input_path_p = '../data/1.keyword_get/keywords/cnc_keywords_multiple_patent.txt'
    input_path_l = '../data/1.keyword_get/keywords/cnc_keywords_multiple_literature.txt'
    node_p = get_keywords(input_path_p)
    node_l = get_keywords(input_path_l)
    # 统计
    print('num of kw in p:', len(node_p))
    print('num of kw in l:', len(node_l))
    print('num of kw common:', len(list(set(node_p) & set(node_l))))
    print('num of kw all:', len(list(set(node_p) | set(node_l))))
    # 输出
    df_p = pd.DataFrame(node_p)
    df_l = pd.DataFrame(node_l)
    output_path = '../data/1.keyword_get/keywords/keyword.xlsx'
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
        df_p.to_excel(writer, sheet_name='patent', header=None)
        df_l.to_excel(writer, sheet_name='literature', header=None)
