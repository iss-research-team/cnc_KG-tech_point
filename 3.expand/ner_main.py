#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/2/28 上午10:42
# @Author  : liu yuhan
# @FileName: ner_main.py
# @Software: PyCharm

import sys

from ner_model import *
from ner_utils_plus import *
from ner_parser import *

from torchsummary import summary


def train():
    # parameter
    args = parameter_parser()

    # model save
    model_save_path = '../data/3.expend/ner_model'
    make_path(model_save_path)
    # 生成数据+数据预处理
    with open('../data/3.expend/test_input/ner_dataset_public.json', 'r', encoding='UTF-8') as file:
        data_list = json.load(file)
    with open('../data/3.expend/test_input/ner_tag_dict_public.json', 'r', encoding='UTF-8') as file:
        tag_dict = json.load(file)
    tok_list, seg_list, mask_list, tag_list, lens_list = make_data(data_list[:100], tag_dict,
                                                                   args.max_len, args.num_class)
    loader_train = Data.DataLoader(MyDataSet(tok_list, seg_list, mask_list, tag_list, lens_list), args.batch_size, True)
    tok_list, seg_list, mask_list, tag_list, lens_list = make_data(data_list[100:124], tag_dict,
                                                                   args.max_len, args.num_class)
    loader_eval = Data.DataLoader(MyDataSet(tok_list, seg_list, mask_list, tag_list, lens_list), args.batch_size, True)
    print('loader done.')
    # 载入模型、优化器
    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    my_ner = MyNER(args.batch_size, args.num_tag, args.max_len, args.embed_dim,
                   args.lstm_dim, args.lstm_layers, args.lstm_dropout, args.dropout, args.if_load_pretrain).to(device)
    # 打印模型结构
    summary(my_ner)

    optimizer = optim.Adam(my_ner.parameters(), lr=0.0001)

    for epoch in tqdm(range(args.epochs)):
        # train
        my_ner.train()
        loss_collector = []
        for i, data in enumerate(loader_train):
            # 数据载入
            tokens_tensor, segments_tensor, mask_tensor, tag_tenser, lens_list = data
            # forward和loss
            feats = my_ner(tokens_tensor=tokens_tensor.to(device),
                           segments_tensor=segments_tensor.to(device),
                           attention_mask=mask_tensor.to(device),
                           lens=lens_list.to(device))
            loss = my_ner.loss_fun(feats,
                                   attention_mask=mask_tensor.to(device),
                                   tag_tenser=tag_tenser.to(device))
            loss.backward()
            optimizer.step()
            loss_collector.append(loss.item())
            if i % 50 == 0:
                print(i, 'loss:', loss.item())
        loss_mean = np.mean(loss_collector)

        # eval
        my_ner.eval()
        acc_collector = []
        for i, data in enumerate(loader_eval):
            # 数据载入
            tokens_tensor, segments_tensor, mask_tensor, tag_tenser, lens_list = data
            # forward和loss
            feats = my_ner(tokens_tensor=tokens_tensor.to(device),
                           segments_tensor=segments_tensor.to(device),
                           attention_mask=mask_tensor.to(device),
                           lens=lens_list.to(device))
            pre_tenser = my_ner.predict(feats,
                                        attention_mask=mask_tensor.to(device))

            acc = get_accuracy(pre_tenser, tag_tenser.tolist(), lens_list.tolist())
            acc_collector.append(acc)
        acc_mean = np.mean(acc_collector)
        print('epoch', epoch, 'loss:', loss_mean, 'acc:', acc_mean)


if __name__ == '__main__':
    train()
