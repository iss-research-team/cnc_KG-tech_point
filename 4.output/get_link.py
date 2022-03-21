#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/19 上午2:35
# @Author  : liu yuhan
# @FileName: get_link.py
# @Software: PyCharm
# 获取科技文献和关键技术的连接

import json
from collections import defaultdict
from tqdm import tqdm


def get_link(label):
    with open('../data/1.keyword_get/cnc_doc_' + label + '2index.json', 'r', encoding='UTF-8') as file:
        doc2index = json.load(file)
    with open('../data/2.layer_get/keyword2index.json', 'r', encoding='UTF-8') as file:
        keyword2index = json.load(file)

    doc_keyword = defaultdict(set)

    for doc, doc_index in tqdm(doc2index.items()):
        for keyword, keyword_index in keyword2index.items():
            if ' ' + keyword + ' ' in ' ' + doc + ' ':
                doc_keyword[doc_index].add(keyword_index)

    doc_keyword = dict((doc, list(keyword_set)) for doc, keyword_set in doc_keyword.items())

    with open('../data/4.output/doc_' + label + '_keyword_dict.json', 'w', encoding='UTF-8') as file:
        json.dump(doc_keyword, file)


if __name__ == '__main__':
    label = 'literature'
    get_link(label)
