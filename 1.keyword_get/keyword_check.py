#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/17 下午12:25
# @Author  : liu yuhan
# @FileName: keyword_check.py
# @Software: PyCharm
"""
这个文件用于对获取的关键词进行检查
"""

import json

if __name__ == '__main__':
    with open('../data/1.keyword_get/cnc_keywords_patent.json') as file:
        kw_list_p = json.load(file)
    with open('../data/1.keyword_get/cnc_keywords_literature.json') as file:
        kw_list_l = json.load(file)

    print('num of kw in p:', len(kw_list_p))
    print('num of kw in l:', len(kw_list_l))
    common_list = list(set(kw_list_p) & set(kw_list_l))
    print('num of kw common:', len(common_list))
