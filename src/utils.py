#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/12/2 11:35
@File:utils.py
@Desc:共享工具函数
"""
# 1.利用Pandas构建事件表，同步时间管理
import logging
import os
import pandas as pd
import time
import threading
import numpy as np
import networkx as nx
import re
from datetime import datetime
import matplotlib.pyplot as plt
import collections
import pickle
import random
import multiprocessing as mp
from collections import deque
import pickle
import time
import setting
import json
import copy
import eventlet
eventlet.monkey_patch()
import requests



class Event:
    def __init__(self,event_name,event_time,event_function,event_status):
        """
        定义事件
        :param event_name: 事件名称
        :param event_time: 事件发生的时间
        :param event_function: 事件执行的函数
        """
        self.event_name = event_name
        self.event_time = event_time
        self.event_function = event_function
        self.event_status = event_status


class EventManager:
    def __init__(self):
        # 创建一个空的DataFrame用于存储事件, (Event:事件说明，Timestamp：时间戳, Function：事件执行代码，Status:时间状态
        # （Pending:等待执行，Processed:处理完成）)
        self.events = pd.DataFrame(columns=['Event','Timestamp','Function','Status'])

    def add_event(self,event): # event:Event类
        event_name = event.event_name
        event_time = event.event_time
        event_function = event.event_function
        event_status = event.event_status
        new_event = pd.DataFrame([[event_name,event_time,event_function,event_status]],
                    columns = self.events.columns)

        # 将事件添加到DataFrame中
        self.events = pd.concat([self.events,new_event],ignore_index=True)
        # TODO 输出事件日志 print(f"事件 '{event_name}' 已添加，时间: {timestamp}。")

    def process_events(self):
        # 事件处理 -- 未处理的时间标记为Pending,处理后的数据标记为Processed
        pending_events = self.events[self.events['Status'] == 'Pending']
        if not pending_events.empty:
            # 按Timestamp 列升序排列
            pending_events = pending_events.sort_values(by='Timestamp')
            return self.handle_event(pending_events[0:1])

    # 处理单个事件
    def handle_event(self,event):
        # TODO 输出任务处理日志 print(f"处理事件: '{event_name}'，时间: {self.events.loc[index, 'Timestamp']}")
        # 运行事件功能函数
        print(event['Function'].values)
        # eval(event['Function'].values[0])

        # 更新时间的状态
        self.events.at[0,'Status'] = 'Processed'
        return event['Function'].values[0]
        # TODO 输出事件处理完成日志  print(f"事件 '{event_name}' 处理完成。")

# 日志管理系统
def log_management(file_name,print_level=logging.INFO,file_level=logging.DEBUG):
    # 格式化当前时间用于文件名
    formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    # 日志格式：时间 - 级别 - 消息
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S %p")

    # 创建 logger，设置最低等级为 DEBUG，确保捕获所有日志
    logger = logging.getLogger('run_logger')
    logger.setLevel(logging.DEBUG)

    # 清理已有 handlers，避免重复添加
    if logger.hasHandlers():
        logger.handlers.clear()

    # 确保日志目录存在
    log_dir = setting.logging_path
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建文件处理器，写入所有级别日志（DEBUG及以上）
    file_handler = logging.FileHandler(f'{log_dir}/{file_name}_run_logger_{formatted_time}.log')
    file_handler.setLevel(file_level)  # 文件日志最低级别
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    # 不添加控制台 handler，即不输出日志到控制台
    # 添加打印日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(print_level)
    # 设置打印格式
    console_handler.setFormatter(log_format)

    # 添加打印日志
    logger.addHandler(console_handler)

    # 写一条启动日志
    logger.critical(f'[{file_name}.py]_run_logger_{formatted_time}')

    return logger


# 绘制拓扑
def graph_plot(graph):
    """
    绘制topo图
    :param graph:
    :return:
    """
    pos = nx.spring_layout(graph, seed=42)  # 定义节点位置
    node_color = []
    for node in graph.nodes:  # 将主机和路由器区分颜色
        if 'sta' in node:
            node_color.append('r')
        if 'ap' in node:
            node_color.append('b')
    nx.draw(graph, pos, with_labels=True, node_size=500,
            node_color=node_color, font_size=10, font_weight="bold")
    edge_labels = nx.get_edge_attributes(graph, 'weight')  # 获取权重
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)  # 将权重绘制到图中
    # topo_name = re.split(r'[\./]', self.xml_path)[-2]
    # # print(re.split(r'[\./]',self.xml_path)[-2])
    # plt.savefig(f'../data/figure/{topo_name}_{datetime.now().strftime("%Y-%m-%d")}.png')
    plt.show()


# 经验缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 设置缓冲区
        self.lock = threading.Lock()  # 互斥访问
        self.not_empty = threading.Condition(self.lock)  # 等待数据到达条件


    def add(self,*experience):
        # print('[添加样本]',experience)
        with self.lock:  # 进入临界区
            self.buffer.append(experience)  # 添加数据
            self.not_empty.notify_all()  # 唤醒等待的消费者

    def sample(self, batch_size):
        with self.not_empty:  # 进入条件等待区
            while len(self.buffer) < batch_size:
                self.not_empty.wait()  # 进入阻塞状态
            return random.sample(self.buffer, batch_size)


    def size(self):
        with self.lock:  # 安全读取
            return len(self.buffer)


    def save(self,filename):   # 保存
        with open(filename,'wb') as f:
            pickle.dump(self.buffer,f)


    def load(self,filename):   # 读取
        with open(filename,'rb') as f:
            self.buffer = pickle.load(f)


import random
from multiprocessing.managers import SyncManager


# 直接传递共享对象到子进程
def worker(arr, idx, size, lock):
    # 需要重新包装为可调用对象
    local_buf = LightweightBuffer(1000)
    local_buf._array = arr
    local_buf._index = idx
    local_buf._size = size
    local_buf._lock = lock

    # 执行操作
    for i in range(100):
        local_buf.add((i,i+1))
    print(local_buf.sample(10))


class LightweightBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self._array = mp.Array('d', capacity)
        self._index = mp.Value('i', 0)
        self._size = mp.Value('i', 0)
        self._lock = mp.Lock()

    def add(self, value):
        with self._lock:
            idx = self._index.value % self.capacity
            self._array[idx] = value
            self._index.value += 1
            self._size.value = min(self._size.value + 1, self.capacity)

    def sample(self, batch_size):
        with self._lock:
            current_size = self._size.value
            if current_size == 0:
                return []
            indices = random.sample(range(current_size),
                                    min(batch_size, current_size))
            base = self._index.value - current_size
            return [self._array[(base + i) % self.capacity]
                    for i in indices]

# 读取最新模型文件名
def read_last_weight_file(path,identifier=None):
    """
    读取最新的权重文件，两层文件
    :param path:
    :return:
    """
    try:
        # 遍历目录
        entries = list(os.scandir(path))
        if identifier:
            entries = [item for item in entries if identifier in item.name]
        if not entries:
            return None
        # 用修改时间作为key，找到最大值的文件
        lastest = max(entries,key=lambda x: x.stat().st_mtime)
        if not lastest.is_file():
            return read_last_weight_file(lastest.path,identifier=identifier)
        else:
            return lastest.path

    except FileNotFoundError:
        return None  # 处理目录不存在的情况


def calculate_node_order(G,cur_site,dst,weight_list):
    """
    计算节点的序,与当前节点无关，不用每次都进行计算，计算后查表即可  TODO 进行优化
    :param G: 图
    :param cur_site: 当前节点的位置，用于修正排序
    :param dst: 目的节点
    :param weight_list: 暂时只支持单个权重
    :return:
    """
    if isinstance(weight_list,list):
        weight = weight_list[0]
    else:
        weight = weight_list
    # 计算每对节点之间的最短路径的权重和
    shortest_path_matrix = np.zeros((len(G.nodes),len(G.nodes)))

    # 遍历每对节点
    for index1,u in enumerate(G.nodes()):
        for index2,v in enumerate(G.nodes()):
            if u != v:
                # 获取 u 和 v 之间的最短路径（使用 Dijkstra 算法）
                path = nx.dijkstra_path(G, source=u, target=v, weight=weight)
                path_weight_sum = sum(G[u][v][weight] for u, v in zip(path[:-1], path[1:]))
                shortest_path_matrix[index1,index2] = path_weight_sum
            else:
                shortest_path_matrix[index1,index2] = 0

    # # 打印每对节点之间的最短路径的权重和
    print('节点顺序',list(G.nodes))
    # print('最短路径值',shortest_path_matrix[:,list(G.nodes).index(dst)])
    shortest_path_value = shortest_path_matrix[:,list(G.nodes).index(dst)].ravel()


    # 绑定列表进行排序
    zipped = list(zip(list(G.nodes),shortest_path_value))
    # print(f'目的节点{dst}，其他节点的值{shortest_path_value}')
    # ziped_sorted = sorted(zipped,key=lambda x: (x[1],x[0]==cur_site),reverse=True)  # 修正排序，存在多个节点序一样时，将当前的节点放在最前面
    ziped_sorted = sorted(zipped, key=lambda x: x[1], reverse=True)  # 修正也可能产生回路
    sorted_list1, sorted_list2 = zip(*ziped_sorted)

    # print(sorted_list1)
    # if dst == 17:
    # print(f'目标节点{dst}的排序值{sorted_list2}')
    # print(f'目标节点{dst}的序{sorted_list1}')
    return sorted_list1, sorted_list2

# 1. 定义发送类
class SendData:
    def __init__(self,server_ip, server_port):
        self.server_ip = server_ip    # 服务端ip
        self.server_port = server_port   # 服务端端口


    async def send_data(self, data, server_path):
    # def send_data(self, data, server_path):
        """
        在图属性的基础上添加路径的标识
        :param data:
        :param server_path:
        :return:
        """
        # 2.构建数据
        response = requests.post(
            f'http://{self.server_ip}:{self.server_port}' + server_path,
            json=data,
            headers={'Content-Type': 'application/json'}
        )

class Evaluate:
    def __init__(self, save_figure_path='../data/result/figure/',save_data_path='../data/result/data/'):
        """
        评价类
        :param save_figure_path: 保存图片的路径
        :param save_data_path: 保存数据的路径
        """
        self.save_figure_path = save_figure_path
        self.save_data_path = save_data_path

    def data_display(self, eva_count_list, eva_value_list, tag_name, disp_mode):
        """
        显示评价结果
        :param eva_count_list: 评价对应的代数列表
        :param eva_value_list: 评价的值列表，上下层不一样，上层是值列表，下层是字典列表，分两种情况进行处理
        :param tag_name: 标识性名字，用于区分不同的绘图，file_name+次数
        :param disp_mode: 模式 "H" or "L"
        :return: 保存三个数据和图片
        """
        if disp_mode == 'H':
            prefix = 'High_'  # 按上层数据进行处理
            # 将数据转化为pandas
            data = {
                'eva_count_list': eva_count_list,
                'eva_value_list': eva_value_list
            }
            df = pd.DataFrame(data)

            title_name = 'eva_value_data_'

            title_name1 = 'eva_value_over_count_range_'
            plt.figure()
            df.plot(x='eva_count_list', y='eva_value_list', title=title_name1)
            plt.savefig(f'{self.save_figure_path}{prefix}{title_name1}'
                        f'{setting.high_actor_lr}_{setting.high_critic_lr}_{setting.low_actor_lr}_{setting.low_critic_lr}_{setting.high_alpha_lr}_'
                        f'{tag_name}.png')
            plt.show()
            df.to_csv(f'{self.save_data_path}{prefix}{title_name}'
                      f'{setting.high_actor_lr}_{setting.high_critic_lr}_{setting.low_actor_lr}_{setting.low_critic_lr}_{setting.high_alpha_lr}_'
                      f'{tag_name}.csv', index=False,
                      encoding='utf-8-sig')  # 支持Excel打开

        elif disp_mode == 'L':
            prefix = 'Low_'  # 按下层数据进行处理
            df = pd.DataFrame(eva_value_list)
            df['eva_count_list'] = eva_count_list
            title_name = 'eva_value_data_'
            for key in eva_value_list[0].keys():
                title_name2 = f'eva_value_over_count_list_{key}_'
                plt.figure()
                df.plot(x='eva_count_list', y=key, title=title_name2)
                plt.savefig(f'{self.save_figure_path}{prefix}{title_name2}'
                            f'{setting.high_actor_lr}_{setting.high_critic_lr}_{setting.low_actor_lr}_{setting.low_critic_lr}_{setting.high_alpha_lr}_'
                            f'{tag_name}.png')
                plt.show()
            df.to_csv(f'{self.save_data_path}{prefix}{title_name}'
                      f'{setting.high_actor_lr}_{setting.high_critic_lr}_{setting.low_actor_lr}_{setting.low_critic_lr}_{setting.high_alpha_lr}_'
                      f'{tag_name}.csv', index=False,
                      encoding='utf-8-sig')  # 支持Excel打开

        else:
            print(f'[Evalue]暂时不支持这种类型{disp_mode}，请检查。。。')

        # if eva_count_range == []:
        #     prefix = 'High_'  # 按上层数据进行处理
        #     # 将数据转化为pandas
        #     data = {
        #         'eva_count_range': eva_count_list,
        #         'eva_value_list': eva_value_list
        #     }
        #     df = pd.DataFrame(data)
        #
        #     title_name = 'eva_value_data_'
        #
        #     title_name1 = 'eva_value_over_count_range'
        #     plt.figure()
        #     df.plot(x='eva_count_range', y='eva_value_list', title=title_name1)
        #     plt.savefig(f'{self.save_figure_path}{prefix}{title_name1}{tag_name}')
        #     plt.show()
        #     df.to_csv(f'{self.save_data_path}{prefix}{title_name}{tag_name}.csv', index=False,
        #               encoding='utf-8-sig')  # 支持Excel打开
        # elif isinstance(eva_value_list[0], float):
        #     prefix = 'High_'         # 按上层数据进行处理
        #     # 将数据转化为pandas
        #     data = {
        #         'eva_count_range': eva_count_range,
        #         'eva_count_list': eva_count_list,
        #         'eva_value_list': eva_value_list
        #     }
        #     df = pd.DataFrame(data)
        #
        #     title_name = 'eva_value_data_'
        #
        #     title_name1 = 'eva_value_over_count_range_'
        #     plt.figure()
        #     df.plot(x='eva_count_range', y='eva_value_list', title=title_name1)
        #     plt.savefig(f'{self.save_figure_path}{prefix}{title_name1}{tag_name}')
        #     plt.show()
        #     title_name2 = 'eva_value_over_count_list_'
        #     plt.figure()
        #     df.plot(x='eva_count_list', y='eva_value_list', title=title_name2)
        #     plt.savefig(f'{self.save_figure_path}{prefix}{title_name2}{tag_name}')
        #     plt.show()
        #
        #     df.to_csv(f'{self.save_data_path}{prefix}{title_name}{tag_name}.csv',index=False,encoding='utf-8-sig')   # 支持Excel打开

        # elif isinstance(eva_value_list[0], dict):
        #     prefix = 'Low_'           # 按下层数据进行处理
        #     df = pd.DataFrame(eva_value_list)
        #     df['eva_count_range'] = eva_count_range
        #     df['eva_count_list'] = eva_count_list
        #     title_name = 'eva_value_data_'
        #     for key in eva_value_list[0].keys():
        #         title_name1 = f'eva_value_over_count_range_{key}_'
        #         plt.figure()
        #         df.plot(x='eva_count_range', y=key, title=title_name1)
        #         plt.savefig(f'{self.save_figure_path}{prefix}{title_name1}{tag_name}')
        #         plt.show()
        #         df.to_csv(f'{self.save_data_path}{prefix}{title_name1}{tag_name}.csv', index=False,
        #                   encoding='utf-8-sig')  # 支持Excel打开
        #         title_name2 = f'eva_value_over_count_list_{key}_'
        #         plt.figure()
        #         df.plot(x='eva_count_list', y=key, title=title_name2)
        #         plt.savefig(f'{self.save_figure_path}{prefix}{title_name2}{tag_name}')
        #         plt.show()
        #     df.to_csv(f'{self.save_data_path}{prefix}{title_name}{tag_name}.csv', index=False,
        #               encoding='utf-8-sig')  # 支持Excel打开
        #
        # else:
        #     print(f'[Evalue]暂时不支持这种数据{eva_value_list}，请检查。。。')

def weighted_random(pro_dict):
    """
    根据概率权重选择目标主机
    :param pro_dict: 权重字典
    :return:
    """
    rand_val = random.random()   # 生成0-1之间的随机数
    cumulative = 0
    if isinstance(pro_dict,dict):
        for key, weight in pro_dict.items():
            cumulative += weight
            if rand_val <= cumulative:
                return key
        return list(pro_dict.keys())[-1]  # 处理浮点进度问题
    elif isinstance(pro_dict,list):
        keys = list(setting.choice_pro.keys())
        for key, weight in zip(keys, pro_dict):
            cumulative += weight
            if rand_val <= cumulative:
                return key
        return keys[-1]
    else:
        print(f"[action]出现了不支持的数据类型{pro_dict}")
        return setting.choice_pro.keys()[-1]


def exp_init(tag):
    """
    初始化实验路径，用于区分不同实验结果
    :param tag: 结果标识
    :return:
    """
    setting.tag = tag
    # 存储路径
    setting.model_path = f'../data/result{tag}/models/'  # 模型
    setting.save_figure_path = f'../data/result{tag}/figure/'  # 图片
    setting.save_data_path = f'../data/result{tag}/data/'  # 数据
    path_list = [setting.model_path,setting.save_figure_path,setting.save_data_path]
    for path in path_list:
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path, exist_ok=True)

def smooth(data,sm=10):
    """
    平滑函数，前sm个数据求均值，后面每sm个求一次均值
    :param data:
    :param sm:
    :return:
    """
    smooth = []
    if isinstance(data,list):
        for ind,d in enumerate(data):
            if ind<sm:
                print(ind,d)
                smooth.append(sum(data[:(ind+1)])/(ind+1))
            else:
                print(ind, d)
                smooth.append(sum(data[(ind-sm+1):(ind+1)])/sm)
    return smooth





# 使用示例
if __name__ == '__main__':
    # buffer = LightweightBuffer(1000)
    #
    # p = mp.Process(target=worker,
    #                args=(buffer._array, buffer._index,
    #                      buffer._size, buffer._lock))
    # p.start()
    # p.join()
    # import networkx as nx
    #
    # # 创建图
    # G = nx.Graph()
    #
    # # 添加带有多个属性的边
    # G.add_edge(1, 2, weight=5, cost=3, type='A')
    # G.add_edge(2, 3, weight=2, cost=4, type='B')
    # G.add_edge(3, 4, weight=8, cost=1, type='A')
    # G.add_edge(1, 4, weight=6, cost=2, type='C')
    #
    # src = 1
    # dst = 3
    # weight_list = 'cost'
    # calculate_node_order(G, src, dst, weight_list)
    # tag = 123
    # exp_init(tag)
    data_list = list(range(10))
    smooth_data = smooth(data_list,5)
    print(smooth_data)