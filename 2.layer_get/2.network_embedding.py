#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 下午3:56
# @Author  : liu yuhan
# @FileName: 2.network_embedding.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json
import sys
import os
from tqdm import tqdm


def not_empty(s):
    return s and s.strip()


# loss曲线绘制
def loss_draw(epochs, loss_list, loss_save_path):
    plt.plot([i + 1 for i in range(epochs)], loss_list)
    plt.savefig(loss_save_path)


class NetworkDeal():
    '''
    生成网络，
    '''

    def __init__(self, node_path, link_path, ng_num):
        self.ng_num = ng_num
        with open(node_path, 'r', encoding='UTF-8') as file:
            self.node_list = json.load(file)
        self.node_num = len(self.node_list)
        with open(link_path, 'r', encoding='UTF-8') as file:
            self.link_list_weighted = json.load(file)
        print('node_num:', self.node_num)
        print('link_num:', len(self.link_list_weighted))

    def get_network_feature(self):
        '''
        构建网络计算个重要参数——degree
        :return:
        '''
        g = nx.Graph()
        g.add_nodes_from([i for i in range(self.node_num)])
        g.add_weighted_edges_from(self.link_list_weighted)
        self.node_degree_dict = dict(nx.degree(g))

    def get_data(self):
        '''
        负采样
        :return:
        '''
        # 负采样
        sample_table = []
        sample_table_size = 1e8
        # 词频0.75次方
        pow_frequency = np.array(list(self.node_degree_dict.values())) ** 0.75
        nodes_pow = sum(pow_frequency)
        ratio = pow_frequency / nodes_pow
        count = np.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            sample_table += [wid] * int(c)
        sample_table = np.array(sample_table)

        # 生成一个训练集：
        source_list = []
        target_list = []
        node_ng_list = []

        for link in self.link_list_weighted:
            source_list.append(link[0])
            target_list.append(link[1])
            node_ng = np.random.choice(sample_table, size=(self.ng_num)).tolist()
            node_ng_list.append(node_ng)

        return self.node_num, \
               torch.LongTensor(source_list), torch.LongTensor(target_list), torch.LongTensor(node_ng_list)


class Line(nn.Module):
    def __init__(self, word_size, dim, node_feature_path: str = ''):
        super(Line, self).__init__()
        initrange = 0.5 / dim
        # input
        if node_feature_path:
            weight = np.load(node_feature_path, encoding="latin1")
            weight = torch.from_numpy(weight).float()
            self.u_emd = nn.Embedding(word_size, dim, _weight=weight)
            self.context_emd = nn.Embedding(word_size, dim, _weight=weight)

        else:
            self.u_emd = nn.Embedding(word_size, dim)
            self.u_emd.weight.data.uniform_(-initrange, initrange)
            self.context_emd = nn.Embedding(word_size, dim)
            self.context_emd.weight.data.uniform_(-0, 0)

    # 这边进行一个一阶+二阶的
    def forward(self, s, t, ng):
        vector_i = self.u_emd(s)
        # 一阶
        vector_o1 = self.u_emd(t)
        vector_ng1 = self.u_emd(ng)
        output_1_1 = torch.matmul(vector_i, vector_o1.transpose(-1, -2)).squeeze()
        output_1_1 = F.logsigmoid(output_1_1)
        # 负采样的部分
        output_1_2 = torch.matmul(vector_i.unsqueeze(1), vector_ng1.transpose(-1, -2)).squeeze()
        output_1_2 = F.logsigmoid(-1 * output_1_2).sum(1)

        output_1 = -1 * (output_1_1 + output_1_2)
        # 二阶
        vector_o2 = self.context_emd(t)
        vector_ng2 = self.context_emd(ng)
        output_2_1 = torch.matmul(vector_i, vector_o2.transpose(-1, -2)).squeeze()
        output_2_1 = F.logsigmoid(output_2_1)
        # 负采样的部分
        output_2_2 = torch.matmul(vector_i.unsqueeze(1), vector_ng2.transpose(-1, -2)).squeeze()
        output_2_2 = F.logsigmoid(-1 * output_2_2).sum(1)

        # 组合
        output_2 = -1 * (output_2_1 + output_2_2)

        loss = torch.mean(output_1) + torch.mean(output_2)

        return loss

    # 保存参数
    def save_embedding(self, file_name):
        """
        Save all embeddings to file.
        """
        embedding = self.u_emd.weight.cpu().data.numpy()
        np.save(file_name, embedding)


class MyDataSet(Data.Dataset):
    '''
    没啥说的，正常的数据载入
    '''

    def __init__(self, s_list, t_list, ng_list):
        self.s_list = s_list
        self.t_list = t_list

        self.ng_list = ng_list

    def __len__(self):
        return len(self.s_list)

    def __getitem__(self, idx):
        return self.s_list[idx], self.t_list[idx], self.ng_list[idx]


if __name__ == '__main__':
    # 参数设置
    d = 512
    batch_size = 32
    epochs = 2

    label = sys.argv[1]
    node_path = "../data/1.keyword_get/cnc_keywords_" + label + ".json"
    link_path = "../data/2.get_layer/cnc_keywords_link_" + label + ".json"

    ng_num = 5

    node_feature_path = "../data/2.get_layer/node_emb_word_" + label + ".npy"
    node_emb_path = "../data/2.get_layer/node_emb_net_" + label
    if not os.path.exists(node_emb_path):
        os.mkdir(node_emb_path)

    # 数据处理
    networkdeal = NetworkDeal(node_path, link_path, ng_num)
    networkdeal.get_network_feature()
    node_size, s_list, t_list, ng_list = networkdeal.get_data()
    loader = Data.DataLoader(MyDataSet(s_list, t_list, ng_list), batch_size, True)

    # 模型初始化
    line = Line(node_size, d, node_feature_path)
    line.cuda()
    optimizer = optim.Adam(line.parameters(), lr=0.0001)

    # 保存平均的loss
    ave_loss = []

    for epoch in tqdm(range(epochs)):
        loss_collector = []
        for i, (s, t, ng) in enumerate(loader):
            s = s.cuda()
            t = t.cuda()
            ng = ng.cuda()
            loss = line(s, t, ng)
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print("epoch", epoch, "loss", loss.item())
            loss_collector.append(loss.item())
        ave_loss.append(np.mean(loss_collector))
        if epoch > 0 and epoch % 10 == 0:
            line.save_embedding(os.path.join(node_emb_path, "epoch_" + str(epoch)))
    loss_save_path = '../data/fig/embed_loss_' + label + '.png'
    loss_draw(epochs, ave_loss, loss_save_path)
