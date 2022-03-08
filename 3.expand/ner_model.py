#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/2/28 上午10:41
# @Author  : liu yuhan
# @FileName: ner_model.py
# @Software: PyCharm


import torch
import ipdb
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from torch.autograd import Variable
from torch.nn.utils import rnn
from transformers import BertModel
from torchcrf import CRF
from sklearn.metrics import precision_score, recall_score, f1_score


class MyDataSet(Data.Dataset):
    '''
    没啥说的，正常的数据载入
    '''

    def __init__(self, token_list, segment_list, mask_list, target_list, lens_list):
        self.token_list = token_list
        self.segment_list = segment_list
        self.mask_list = mask_list
        self.target_list = target_list
        self.lens_list = lens_list

    def __len__(self):
        return len(self.token_list)

    def __getitem__(self, idx):
        return self.token_list[idx], self.segment_list[idx], self.mask_list[idx], self.target_list[idx], \
               self.lens_list[idx]


class MyNER(nn.Module):
    def __init__(self, batch_size, num_tag, max_len, embed_dim, lstm_dim, lstm_layers, lstm_dropout, dropout):
        super(MyNER, self).__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.lstm_dim = lstm_dim
        self.lstm_layers = lstm_layers
        # bert config
        self.word_embeds = BertModel.from_pretrained('bert-base-uncased')
        # Bilstm
        self.lstm = torch.nn.LSTM(input_size=embed_dim, hidden_size=lstm_dim, num_layers=lstm_layers,
                                  batch_first=True, dropout=lstm_dropout, bidirectional=True)
        self.hidden = self.rand_init_hc()
        self.cell = self.rand_init_hc()
        # linear
        self.dropout = nn.Dropout(p=dropout)
        self.linear_lstm = torch.nn.Sequential(torch.nn.Linear(in_features=2 * lstm_dim, out_features=lstm_dim),
                                               torch.nn.ReLU())
        # linear-crf
        self.linear_crf = torch.nn.Sequential(torch.nn.Linear(in_features=lstm_dim, out_features=num_tag),
                                              torch.nn.ReLU())
        self.crf = CRF(num_tag, batch_first=True)

    def rand_init_hc(self):
        """
        random initialize hidden variable
        """
        return Variable(torch.randn(2 * self.lstm_layers, self.batch_size, self.lstm_dim)).cuda()

    def forward(self, tokens_tensor, segments_tensor, attention_mask, lens):
        # BERT
        embeds = self.word_embeds(tokens_tensor, segments_tensor, attention_mask)[0]
        # Bilstm
        # batch_packed = rnn.pack_padded_sequence(input=embeds, lengths=lens.cpu(), batch_first=True,
        #                                         enforce_sorted=False)
        output, _ = self.lstm(embeds)
        # output = rnn.pad_packed_sequence(output, batch_first=True, total_length=self.max_len)[0]
        # linear
        output = self.dropout(output)
        output = self.linear_lstm(output)
        # linear-crf
        output = self.linear_crf(output)
        return output

    def loss_fun(self, feats, attention_mask, tag_tenser):
        # loss
        loss = -1 * self.crf(feats, tag_tenser, mask=attention_mask.byte())
        return loss

    def predict(self, feats, attention_mask):
        # predict
        pre_tenser = self.crf.decode(feats, mask=attention_mask.byte())
        return pre_tenser
