#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2025/4/8 16:35
@File:setting.py
@Desc:****************
"""
import torch

# 属性指标即顺序
# edge_attr_choice = ['free_bw', 'delay', 'loss', 'distance', 'forward_queue_pkts']
edge_attr_choice = ['free_bw', 'delay', 'loss', 'forward_queue_pkts']

# 反向指标序列
# rev_edge_attr_choice = ['free_bw', 'delay', 'loss', 'distance', 'reverse_queue_pkts']
rev_edge_attr_choice = ['free_bw', 'delay', 'loss', 'reverse_queue_pkts']

edge_attr_type = ['MIN', 'FSUM', 'PROD',  'FSUM']
reward_type = ['max', 'min', 'min', 'min']  # 根据优化目标设定奖励的类型

# 比例系数
bate = [0.3, 0.3, 0.2, 0.2]

node_attr_choice = ['cur_site', 'dst_site']
cur_site = node_attr_choice.index('cur_site')

# 源目标序列
src_list = [3]
dst_list = [15, 16, 17, 18]
# src_list = [1]
# dst_list = [5, 6]
# # 上层选择概率
# choice_pro = {
#     '(1, 15)': 0.25,
#     '(1, 16)': 0.25,
#     '(1, 17)': 0.25,
#     '(1, 18)': 0.25,
# }
choice_pro = {}
for src in src_list:
    for dst in dst_list:
        choice_pro.update({f'({src}, {dst})':1/(len(src_list)*len(dst_list))})

# src_dst_pair = ['(1, 15)', '(1, 16)', '(1, 17)', '(1, 18)']
src_dst_pair = [f'({src_list[0]}, {dst})' for dst in dst_list]

# 到达目标的奖励
terminal_reward = 10
# 终端惩罚（未到达目标）
terminal_punishment = -10

# 下一跳位置
node_attr_action = ['next_site']

# 训练参数
# 公共参数
seed = 1234
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 上层参数
train_num_episodes = 800

high_buffer_size = 5000  # 上层buffer的大小
high_buffer_req_size = 16  # 上层buffer的大小要求
high_batch_size = 8  # 上层batch_size
# high_buffer_req_size = 2  # 上层buffer的大小要求
# high_batch_size = 2  # 上层batch_size
# high_actor_lr = 5e-4     # 上层策略网络的学习率
high_critic_lr = 5e-3 * 1    # 上层评价网络的学习率
high_actor_lr = 5e-4 * 2    # 上层策略网络的学习率
# high_critic_lr = 5e-3 * 2   # 上层评价网络的学习率
# high_num_episodes = 200   # 上层迭代次数
high_hid_dim = 16        # 上层隐藏层的维度
high_gamma = 0.98      # 上层折扣因子
high_tau = 0.005       # 上层软更新的比率
high_sigma = 0.01     # 上层高斯标准差
high_dropout = 0.05   # 上层参数丢弃率

# 下层参数
low_buffer_size = 10000
low_buffer_req_size = 32  # 下层buffer的大小要求
low_batch_size = 16  # 下层batch_size
# low_buffer_req_size = 2  # 下层buffer的大小要求
# low_batch_size = 2  # 下层batch_size

# low_actor_lr = 5e-4 * 2    # 下层策略网络的学习率
# low_critic_lr = 5e-3 * 2   # 下层评价网络的学习率
low_actor_lr = 5e-4    # 下层策略网络的学习率
low_critic_lr = 5e-3    # 下层评价网络的学习率
# low_num_episodes = 200   # 下层迭代次数
low_hid_dim = 16        # 下层隐藏层的维度
low_gamma = 0.98      # 下层折扣因子
low_tau = 0.005       # 下层软更新的比率
low_sigma = 0.01     # 下层高斯标准差
low_dropout = 0.05   # 下层参数丢弃率

# 数据服务器ip和端口
# data_server_ip = "10.0.6.101"  # 注意在不同的网络下要进行修改，否者无反应
# # data_server_ip = '10.0.6.101'
data_server_ip = '10.33.32.140'
data_server_port = 5000

# 线程管理
num_collectors = 1
num_learners = 1

tag = f''   # 将在乎的变量写在其中

# 存储路径
model_path = f'../data/result{tag}/models/'  # 模型
save_figure_path = f'../data/result{tag}/figure/'  # 图片
save_data_path = f'../data/result{tag}/data/'  # 数据
result_path = f'../data/result'
logging_path = f'../data/result{tag}/logging/'
log_path = f'../data/result{tag}/log/'

# 定序变量
ordering_attr_list = ['distance']

# 控制变量
newest_model = True  # 要么用最新模型，要么用最优模型

# 控制时间间隔
train_timespan = 1.5
eva_timespan = 2
control_timespan = 5

# 数据服务器的动作路径
data_server_action_path = '/api/action'

# 评价展示间隔
eva_display_interval = 10

# 最大评价步数
max_evaluate_step = 10

# PPO参数
lmbda = 0.95
epochs = 10
eps = 0.2

# 上层模型选择
high_train_model = 'SAC'

# # 上层选择概率
# choice_pro = {
#     '(1, 15)': 0.25,
#     '(1, 16)': 0.25,
#     '(1, 17)': 0.25,
#     '(1, 18)': 0.25,
# }

# 最优控制长度
optional_control = train_num_episodes

# 日志管理
import logging
print_level = logging.INFO
file_level = logging.DEBUG

# SAC补充参数
high_alpha_lr = 1e-4 * 1.4
high_temperature = torch.tensor(1.2)