#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/2 下午8:38
# @Author  : liu yuhan
# @FileName: 0.dataset_make.py
# @Software: PyCharm

import json
from tqdm import tqdm


def get_tech_point():
    """
    排序，按照词语长短
    :return:
    """
    keyword2index_path = '../data/2.layer_get/keyword2index.json'
    with open(keyword2index_path, 'r', encoding='UTF-8') as file:
        keyword2index = json.load(file)
    keyword_list = (keyword2index.keys())
    keyword_list = sorted(keyword_list, key=lambda x: x.count(' '), reverse=True)
    keyword_list = [' ' + keyword + ' ' for keyword in keyword_list]
    return keyword_list


def get_doc():
    doc_path_pat = '../data/1.keyword_get/cnc_doc_patent.txt'
    doc_path_lit = '../data/1.keyword_get/cnc_doc_literature.txt'
    doc_list_pat = open(doc_path_pat, 'r', encoding='UTF-8').readlines()
    doc_list_lit = open(doc_path_lit, 'r', encoding='UTF-8').readlines()
    return doc_list_pat + doc_list_lit


def get_keyword_seq(kw_list, s):
    """
    通过每一句话获取节点的邻居
    :param s:
    :return:
    """
    node_trans_list = []

    for node in kw_list:
        if node not in s:
            continue
        else:
            node_length = node.count(' ') - 1
            bit = 0
            for i in range(s.count(node)):
                bit = s.find(node, bit)
                start_bit = s[:bit].count(' ')
                node_trans = [k for k in range(start_bit, start_bit + node_length)]
                node_trans_list.append(node_trans)
                bit += 1

    return node_trans_list


def overlap_check(node_list):
    """
    时间有限，这边只进行简的check
    :param node_list:
    :return:
    """
    check = True
    num_node = len(node_list)
    for i in range(0, num_node - 1):
        for j in range(i + 1, num_node):
            # 两个子串的index有交集认为是重叠
            if set(node_list[i]) & set(node_list[j]):
                check = False
                break
    return check


def get_target_list(node_list, target_list):
    for node in node_list:
        target_list[node[0]] = "B-tech_point"
        for index in node[1:]:
            target_list[index] = "I-tech_point"
    return target_list


def make_dataset(tech_point_list, doc_list):
    """
    这里的标记用一个小技巧，作用可能有限，但是希望日后有用（和秋静打电话的时候想到的）
    我喜欢白巧克力酱
    分解为
    我喜欢白巧克力
    我喜欢巧克力酱
    :param tech_point_list:
    :param doc_list:
    :return:
    """
    dataset = []
    word_set = set()
    for doc in tqdm(doc_list):
        sentence_list = doc.split('. ')
        for sentence in sentence_list:
            node_trans_list = get_keyword_seq(tech_point_list, ' ' + sentence + ' ')
            if not node_trans_list:
                continue
            # 时间有限，这边只进行简的check
            if overlap_check(node_trans_list):
                source_list = sentence.split()
                target_list = get_target_list(node_trans_list, ["O" for _ in range(len(source_list))])

                # 这里再加一个限制条件，保证训练集的质量
                if 20 > len(source_list) > 10:
                    dataset.append([source_list, target_list])
                    for word in source_list:
                        word_set.add(word)
    # 这里建一个字典
    word_list = sorted(list(word_set))
    word_dict = dict(zip(word_list, [i for i in range(len(word_list))]))
    print(len(word_dict))

    return dataset, word_dict


if __name__ == "__main__":
    tech_point_list = get_tech_point()
    doc_list = get_doc()
    print("data loaded.")
    dataset, word_dict = make_dataset(tech_point_list, doc_list)
    print("dataset size:", len(dataset))
    cut_bit = int(0.7 * len(dataset))

    with open('../data/3.expend/ner_dataset_train.json', 'w', encoding='UTF-8') as file:
        json.dump(dataset[:cut_bit], file)
    with open('../data/3.expend/ner_dataset_test.json', 'w', encoding='UTF-8') as file:
        json.dump(dataset[cut_bit:], file)
    with open('../data/3.expend/word_dict.json', 'w', encoding='UTF-8') as file:
        json.dump(word_dict, file)
