#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/12/2 11:14
@File:hyperparameter_tuning.py
@Desc:超参数调优
"""
#
import datetime

CAPACITY = 10

# 模拟参数
seed = 42  # 随机种子
num_packets = 1000  # 模拟数据包数量
lambda_arrival = 1.5  # 数据包到达的平均速率
mean_size = 1500 * 8   # 数据包大小的均值(bit)
std_size = 300*8  # 数据包大小的标准差(bit)

# 多主机参数
hosts_name = ['h1', 'h2', 'h3', 'h4']
hosts_num_packets = [1000,800,500,900]
hosts_lambda_arrival = [1.5,1.2,0.7,1.4]
hosts_mean_size = [512,800,1500,600]*8
hosts_std_size = [100,150,300,120]*8


# 基线时间
base_time = datetime.datetime(2024,12,1,0,0,0)

# 数据包参数
destination_ip = '10.0.0.255'
protocol = 'TCP'
dateline_time = 60  # s
data = 'XXX'


# 位置参数
lat_min, lat_max = 35, 58
lon_min, lon_max = -20, 20

# 日志管理参数
import logging
print_level = logging.INFO   # 打印水平的参数
file_level = logging.DEBUG   # 文件水平的参数

# topo 参数
xml_path = '../data/topology/topo_node_18.xml'
src_set = {'H1','H2','H3','H4'}
std_set = {'H15','H16','H17','H18'}
