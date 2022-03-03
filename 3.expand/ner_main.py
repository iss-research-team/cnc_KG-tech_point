#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/2/28 上午10:42
# @Author  : liu yuhan
# @FileName: ner_main.py
# @Software: PyCharm

from ner_model import *
from ner_utils import *

from torchsummary import summary


def train():
    # parameter
    epochs = 100
    batch_size = 8
    max_len = 128
    embed_dim = 768
    # bi_lstm parameter
    lstm_dim = 256
    lstm_layers = 6
    dropout_lstm = 0.4
    dropout = 0.2
    # crf parameter
    num_tag = 18
    # model save
    model_save_path = '../data/3.expend/ner_model'
    make_path(model_save_path)
    # 生成数据+数据预处理
    with open('../data/3.expend/ner_dataset_train.json', 'r', encoding='UTF-8') as file:
        data_list_train = json.load(file)
    with open('../data/3.expend/ner_dataset_test.json', 'r', encoding='UTF-8') as file:
        data_list_test = json.load(file)
    with open('../data/3.expend/tag_dict.json', 'r', encoding='UTF-8') as file:
        tag_dict = json.load(file)
    tok_list, seg_list, mask_list, tag_list, lens_list = make_data(data_list_train, tag_dict, max_len)
    loader_train = Data.DataLoader(MyDataSet(tok_list, seg_list, mask_list, tag_list, lens_list), batch_size, True)
    tok_list, seg_list, mask_list, tag_list, lens_list = make_data(data_list_test, tag_dict, max_len)
    loader_eval = Data.DataLoader(MyDataSet(tok_list, seg_list, mask_list, tag_list, lens_list), batch_size, True)
    print('loader done.')
    # 载入模型、优化器
    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    my_ner = MyNER(batch_size, num_tag, max_len, embed_dim,
                   lstm_dim, lstm_layers, dropout_lstm, dropout).to(device)
    # 打印模型结构
    summary(my_ner)

    optimizer = optim.Adam(my_ner.parameters(), lr=0.0001)

    for epoch in tqdm(range(epochs)):
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

            acc = get_accuracy(pre_tenser, tag_tenser)
            acc_collector.append(acc)
        acc_mean = np.mean(acc_collector)
        print('epoch', epoch, 'loss:', loss_mean, 'acc:', acc_mean)


if __name__ == '__main__':
    train()
