#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/19 下午8:07
# @Author  : liu yuhan
# @FileName: 0.word2vec.py
# @Software: PyCharm

#  通过bert获取词语的词向量

import pickle
import json
import torch
import numpy as np
import sys

from transformers import BertTokenizer, BertModel
from tqdm import tqdm


# 构建一个字典并保存 一次性代码
def get_word_dict():
    '''
    构建一个字典，解决token和词语的对应关系
    :return:
    '''
    context = open('vocab.txt', 'r', encoding='UTF-8').readlines()
    context = [word.strip() for word in context]
    word_dict = dict(zip([i for i in range(len(context))], context))
    with open('vocab.pkl', 'wb') as file:
        pickle.dump(word_dict, file)


class WordEmbed():
    def __init__(self, if_load_pretrain, keywords_path, doc_path, save_path):
        self.keywords_path = keywords_path
        self.doc_path = doc_path
        self.save_path = save_path

        with open(self.keywords_path, 'r', encoding='UTF-8') as file:
            self.node_list = json.load(file)
        self.node_dict = dict(zip(self.node_list, [i for i in range(len(self.node_list))]))
        # model
        if if_load_pretrain == "yes":
            self.save_path += '_pretrain'
            self.model_path = '../bert-medium-pretrain'
        else:
            self.save_path += '_origin'
            self.model_path = 'prajjwal1/bert-medium'
        # result
        self.node_embed = np.zeros((len(self.node_list), 512))

    def model_loader(self):
        '''
        载入bert模型
        :return:
        '''
        print('model loading...', self.model_path)
        # 通过词典导入分词器
        self.tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-medium')
        # 通过配置和路径导入模型
        self.bert_model = BertModel.from_pretrained(self.model_path)
        print('model loaded!')

    def get_keyword_temper(self, sentence):
        '''
        获取一个keyword temper，方便获取句子中词语的词向量
        :param sentence:
        :param node_dict:
        :return:
        '''
        node_count = dict()
        for node in self.node_list:
            if node in sentence:
                node_count[node] = sentence.count(node)
        return node_count

    def get_index(self, tokens_tensor, k_tokens_tensor):
        index_list = []
        keywords_length = len(k_tokens_tensor)

        for i in range(1, len(tokens_tensor) - keywords_length):
            if k_tokens_tensor == tokens_tensor[i:i + keywords_length]:
                index_list += [i + l for l in range(keywords_length)]
        # print(index_list)
        return index_list

    def my_bert(self, text, keyword_count):
        '''
        获取一句话中所有的关键词的embedding结果，累加。
        :param text:
        :param keyword_count:
        :return:
        '''
        sen_code = self.tokenizer(text, truncation=True, max_length=512)
        tokens_tensor = torch.tensor([sen_code['input_ids']])
        segments_tensors = torch.tensor([sen_code['token_type_ids']])

        with torch.no_grad():
            encoded_layers = self.bert_model(tokens_tensor, segments_tensors)[0][0].numpy()

        for keyword in keyword_count:
            keyword_emb = np.zeros(512)
            k_tokens_tensor = self.tokenizer.encode(keyword, truncation=True, max_length=512)[1:-1]
            index_list = self.get_index(tokens_tensor[0].tolist(), k_tokens_tensor)
            # print('check', [self.vocab_dict[tokens_tensor[0].tolist()[i]] for i in index_list])
            for index in index_list:
                keyword_emb += encoded_layers[index]
            self.node_embed[self.node_dict[keyword]] += keyword_emb

    def get_bert_feature(self):
        '''
        载入数据，构建网络
        :return:
        '''
        doc_file = open(self.doc_path, 'r', encoding='UTF-8').readlines()
        # 载入bert
        self.model_loader()

        for sentence in tqdm(doc_file):
            node_count = self.get_keyword_temper(sentence)
            if node_count:
                # 在抓取到关键词之后进行下一步操作。
                self.my_bert(sentence, node_count)

    def emb_save(self):
        '''
        引入第二种行归一化的方式。
        :return:
        '''
        self.node_embed = torch.nn.LayerNorm(512)(torch.Tensor(self.node_embed))
        self.node_embed = self.node_embed.detach().numpy()
        # 结果写入
        np.save(self.save_path, self.node_embed)


if __name__ == '__main__':
    label = sys.argv[1]
    if_load_pretrain = sys.argv[2]
    # label = 'patent'
    # if_load_pretrain = 'no'

    keywords_path = '../data/1.keyword_get/keywords/cnc_keywords_' + label + '.json'
    doc_path = '../data/1.keyword_get/doc/cnc_doc_' + label + '.txt'
    result_save_path = '../data/2.layer_get/node_emb_word_' + label
    word_embed = WordEmbed(if_load_pretrain, keywords_path, doc_path, result_save_path)
    word_embed.get_bert_feature()
    word_embed.emb_save()
