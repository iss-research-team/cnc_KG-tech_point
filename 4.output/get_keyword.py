#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 下午4:04
# @Author  : liu yuhan
# @FileName: get_keyword.py
# @Software: PyCharm
import json


def get_keyword():
    index2keyword_path = "../data/2.layer_get/index2keyword.json"
    with open(index2keyword_path, 'r', encoding='UTF-8') as file:
        index2keyword = json.load(file)

    keyword2index = dict([(' | '.join(keyword), index) for index, keyword in index2keyword.items()])

    print(keyword2index)
    keyword2index_path = "../data/4.output/keyword2index_final.json"

    with open(keyword2index_path, 'w', encoding='UTF-8') as file:
        json.dump(keyword2index, file)


if __name__ == '__main__':
    get_keyword()
