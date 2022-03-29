# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # @Time    : 2021/12/6 上午9:58
# # @Author  : liu yuhan
# # @FileName: test.py
# # @Software: PyCharm
#
# from collections import defaultdict
#
# n_dict = {' 1 2 ': 0, ' 1 ': 1, ' 2 ': 2}
# a = ' 1 2 3 1 2 '
#
# node_neighbor_dict_1 = defaultdict(list)
# node_neighbor_dict_2 = defaultdict(list)
#
#
# def get_keyword_seq(s, node_dict):
#     node_list = []
#     node_start_list = []
#     node_length_list = []
#     for node in node_dict:
#         if node in s:
#             node_length = node.count(' ') - 1
#             bit = 0
#             for i in range(s.count(node)):
#                 start_bit = s.find(node, bit)
#                 node_list.append(node_dict[node])
#                 node_start_list.append(s[:start_bit].count(' '))
#                 node_length_list.append(node_length)
#                 bit = start_bit + 1
#         else:
#             continue
#
#     # 获取邻居
#     node_num = len(node_list)
#     s_l = s.count(' ') - 1
#     for i in range(node_num):
#         for j in range(node_num):
#             if i == j:
#                 # 1.避开同一个词
#                 continue
#             if node_list[i] == node_list[j]:
#                 # 2.重复词语的问题
#                 continue
#             # ——————————————————————————————————————————————————————————
#             # 这里更好的设计是通过关键词之间的距离设计一种权重：
#             # A....B
#             # C..B
#             # 在构建A和C之间的连接的时候，考虑ab之间的距离和cb之间的距离不同
#             # ——————————————————————————————————————————————————————————
#             # 计算距离
#             if node_start_list[i] < node_start_list[j]:
#                 continue
#             if node_start_list[i] < node_start_list[j]:
#                 dis = node_start_list[j] - node_start_list[i] - node_length_list[i]
#             else:
#                 dis = node_start_list[i] - node_start_list[j] - node_length_list[j]
#             if dis < 0:
#                 # 3.避开子串、重叠词
#                 continue
#             neighbor_inf = {'node': node_list[j], 'dis': dis, 'ave': s_l / node_num}
#             if j > i:
#                 # 下位邻居
#                 node_neighbor_dict_1[node_list[i]].append(neighbor_inf)
#             else:
#                 # 上位位邻居
#                 node_neighbor_dict_2[node_list[i]].append(neighbor_inf)
#
#
# get_keyword_seq(a, n_dict)
#
# import re
#
#
# pattern_number = re.compile(r'\s[0-9]\s')
# num_trans_dict = {'0': 'zero',
#                        '1': 'one',
#                        '2': 'two',
#                        '3': 'three',
#                        '4': 'four',
#                        '5': 'five',
#                        '6': 'six',
#                        '7': 'seven',
#                        '8': 'eight',
#                        '9': 'nine'}
#
#
# def number_trans(string):
#     """
#     补获文本中的数字，将其置换为英文
#     :return:
#     """
#     string = ' ' + string + ' '
#     number_list = re.findall(pattern_number, string)
#     for number in number_list:
#         string = string.replace(number, ' ' + num_trans_dict[number.strip()] + ' ')
#     return string.strip()
#
#
# a = '5 axis 5'
# print(number_trans(a))

a = """

123

"""

print(a.strip())