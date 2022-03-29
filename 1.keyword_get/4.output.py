#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/8 上午11:19
# @Author  : liu yuhan
# @FileName: 3.combine_check.py
# @Software: PyCharm
"""
clean之后的处理
"""

import json
import pandas as pd

if __name__ == '__main__':
    input_path = '../data/1.keyword_get/keywords/keyword_clean_20220329.xlsx'

    save_path_p = '../data/1.keyword_get/keywords/cnc_keywords_patent.json'
    save_path_l = '../data/1.keyword_get/keywords/cnc_keywords_literature.json'

    node_p = pd.read_excel(input_path, sheet_name='patent', header=None)[0].values.tolist()
    node_l = pd.read_excel(input_path, sheet_name='literature', header=None)[0].values.tolist()

    print('num of kw in p:', len(node_p))
    print('num of kw in l:', len(node_l))
    print('num of kw common:', len(list(set(node_p) & set(node_l))))
    print('num of kw all:', len(list(set(node_p) | set(node_l))))

    # save
    with open(save_path_p, 'w', encoding='UTF-8') as file:
        json.dump(node_p, file)
    with open(save_path_l, 'w', encoding='UTF-8') as file:
        json.dump(node_l, file)
