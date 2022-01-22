import os
import xlrd
import csv
import json


# 0 PT
# 1 AU
# 2 BA
# 3 BE
# 4 GP
# 5 AF
# 6 BF
# 7 CA
# 8 TI 文献标题
# 9 SO 出版物名称 期刊
# 10 SE 丛书标题
# 11 BS
# 12 LA
# 13 DT
# 14 CT
# 15 CY
# 16 CL
# 17 SP
# 18 HO
# 19 DE
# 20 ID
# 21 AB
# 22 C1
# 23 RP
# 24 EM
# 25 RI
# 26 OI
# 27 FU
# 28 FX
# 29 CR
# 30 NR
# 31 TC
# 32 Z9
# 33 U1
# 34 U2
# 35 PU
# 36 PI
# 37 PA
# 38 SN
# 39 EI
# 40 BN
# 41 J9
# 42 JI
# 43 PD
# 44 PY
# 45 VL
# 46 IS
# 47 PN
# 48 SU
# 49 SI
# 50 MA
# 51 BP
# 52 EP
# 53 AR
# 54 DI 有用
# 55 D2 有用
# 56 EA
# 57 PG
# 58 WC
# 59 SC
# 60 GA
# 61 UT
# 62 PM
# 63 OA
# 64 HC
# 65 HP
# 66 DA


def get_str(inf):
    # 保证每一个句子的末尾是“. ”
    if inf:
        if inf[-1] == '.':
            inf += ' '
        else:
            inf += '. '
    return inf


def get_time(EA, PY):
    time = None
    if EA:
        time = int(EA[-4:])
    elif PY:
        time = int(PY)
    return time


def preprocess(prepare_path, json_write_path):
    csv.field_size_limit(500 * 1024 * 1024)

    prepare_file_list = os.listdir(prepare_path)

    literature_dict = dict()
    literature_count = 0

    for each_file in prepare_file_list:
        text_read = open(os.path.join(prepare_path, each_file), 'r', encoding='UTF-16')
        # 跳过前三行
        text_read.readline()
        for each_line in text_read:
            inf_list = each_line.split('\t')
            time = get_time(inf_list[56], inf_list[44])
            if not time:
                continue
            # 文本
            title = get_str(inf_list[8])
            abstract = get_str(inf_list[21])
            key_words_author = inf_list[19]
            key_words_plus = inf_list[20]
            if title in literature_dict:
                continue
            literature_dict[title] = {'time': time,
                                      'doc': title + abstract,
                                      'key_words_author': key_words_author,
                                      'key_words_plus': key_words_plus}

            literature_count += 1

    json.dump(literature_dict, open(json_write_path, 'w', encoding='UTF-8'))
    print('获取文献数量：', literature_count)


if __name__ == '__main__':
    prepare_path = '../../data/1.source_file/literature'
    json_write_path = '../../data/2.processed_file/doc_dict_literature.json'
    preprocess(prepare_path, json_write_path)
