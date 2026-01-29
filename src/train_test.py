#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2025/7/23 15:01
@File:train_test.py
@Desc:****************
"""
import eventlet
eventlet.monkey_patch()
import os
import threading
import queue
import time
import random
import requests
import copy
from utils import ReplayBuffer, log_management, read_last_weight_file,SendData,Evaluate,exp_init,weighted_random
import setting
from data_fraction import State, get_high_action_pro, get_high_reward,take_low_agent_act
from algorithms_model import HighDDPG,LowDDPG,gen_high_act_input_batch,gen_low_act_input_batch,HighPPO,HighSAC
import numpy as np
import torch
import re
import asyncio
import sys
from delete_model_figure_data import delete_data
import math

# 全局变量
CUL_FILE_NAME = os.path.basename(__file__).split('.')[0]  # 当前文件的名称
CUL_TIME_FORMAT = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())  # 当前时间
print(CUL_TIME_FORMAT)
LOGGER = log_management(CUL_FILE_NAME)  # 日志打印结果

class TrainingPipeline:
    def __init__(self):
        # 线程共享缓冲区
        self.high_buffer = ReplayBuffer(setting.high_buffer_size)
        self.low_buffer = ReplayBuffer(setting.low_buffer_size)

        # 线程通信队列
        # self.ctrl_queue = queue.Queue(maxsize=100)
        # self.data_queue = queue.Queue(maxsize=1000)

        # 线程控制标志
        self._running = True
        self.data_server_ip = setting.data_server_ip
        self.data_server_port = setting.data_server_port

        # 初始化模型--仅仅只有训练过程中可以改动，其他情况下需要拷贝
        self.high_agent = None
        self.low_agent = None
        self.data_sender = SendData(setting.data_server_ip,setting.data_server_port)
        self.evaluate_tool = Evaluate(save_figure_path=setting.save_figure_path,save_data_path=setting.save_data_path)

        self.agents_list = []  # 智能体列表
        self.optional_agent = None  # 最优智能体,训练结束之后用
        self.last_agent = None  # 最新智能体，训练期间用

        self.training = True  # 模型训练中的标识
        self.evaluating = True  # 模型评价中的标识

    def data_collector(self):
        """数据采集线程（I/O密集型）"""
        last_timestamp = None
        last_high_state = None
        while True:
            # try:
            response = requests.get(
                f'http://{self.data_server_ip}:{self.data_server_port}/api/state',
                timeout=5
            )
            sample_data = response.json()

            if 'timestamp' not in sample_data.keys():
                print(f'数据{sample_data}时间搓缺失，重新获取数据！')
                time.sleep(1)
                continue
            elif sample_data['timestamp'] == last_timestamp:
                continue

            src_dst_pair = setting.src_dst_pair
            cond_list = []
            for key in src_dst_pair:
                cond_list.append(1 if key in sample_data['edges'][0].keys() else 0)
            if not all(cond_list):
                print("路径信息没收集全，请等待。。。")
                time.sleep(1)
                continue

            try:
                # 处理数据采集逻辑
                # LOGGER.debug(f'sample_data:{sample_data}')  # 到这里数据还正常
                high_state = State(sample_data)
                high_action = get_high_action_pro(sample_data)
                high_reward = get_high_reward(sample_data, high_action)

                if last_high_state is not None:
                    high_experience = (last_high_state, high_action, high_reward, high_state)
                    self.high_buffer.add(high_experience)

                    # 处理下层数据
                    low_samples = high_state.get_low_sample()
                    for low_experience in low_samples:
                        self.low_buffer.add(low_experience)

                last_high_state = copy.deepcopy(high_state)
                last_timestamp = sample_data['timestamp']
            except Exception as e:
                print(f'[采样]数据存在问题{e}，重新采集数据。。。')
                print(sample_data)

            # except Exception as e:
            #     if self._running:
            #         print(f"采集线程异常: {str(e)}")
            #     time.sleep(1)

            if setting.high_train_model == 'SAC':
                high_nfeat_e = len(setting.edge_attr_choice)  # 边特征数量，输入特征数
                high_nhid = setting.high_hid_dim  # 上层隐藏层的维度
                high_nout = len(setting.src_dst_pair)  # 上层动作网络输出特征维数
                high_act_lr = setting.high_actor_lr  # 上层策略网络的学习率
                high_cri_lr = setting.high_critic_lr  # 上层评价网络的学习率
                high_alpha_lr = setting.high_alpha_lr  # 上层系数alpha的学习率
                high_dropout = setting.high_dropout  # 上层参数丢弃率
                high_gamma = setting.high_gamma  # 上层折扣因子
                high_tau = setting.high_tau
                high_target_entropy = -len(setting.src_dst_pair)
                device = setting.device  # 设备

                self.high_agent = HighSAC(high_nfeat_e, high_nhid, high_nout, high_act_lr, high_cri_lr, high_dropout,
                                          high_alpha_lr
                                          , high_target_entropy, high_gamma, high_tau, device)
                # 补充参数
                low_nfeat_e = len(setting.edge_attr_choice)
                low_nfeat_v = len(setting.node_attr_choice)
                low_nhid = setting.low_hid_dim
                low_act_lr = setting.low_actor_lr
                low_cri_lr = setting.low_critic_lr
                low_dropout = setting.low_dropout
                low_gamma = setting.low_gamma
                low_sigma = setting.low_sigma
                low_tau = setting.low_tau
                self.low_agent = LowDDPG(
                    low_nfeat_e, low_nfeat_v, low_nhid,
                    low_act_lr, low_cri_lr, low_dropout, low_gamma,
                    low_sigma, low_tau, device
                )
                run_episodes = 0
                train_num_episodes = setting.train_num_episodes  # 训练代数
                while self._running and run_episodes <= train_num_episodes:
                    if self.high_buffer.size() >= setting.high_buffer_req_size and self.low_buffer.size() >= setting.low_buffer_req_size:
                        high_sample_batch = self.high_buffer.sample(setting.high_batch_size)
                        if high_sample_batch:
                            self.high_agent.update(high_sample_batch)
                            print(f"[训练]上层{run_episodes}次训练结束")
                        low_sample_batch = self.low_buffer.sample(setting.low_batch_size)
                        if low_sample_batch:
                            self.low_agent.update(low_sample_batch)
                            print(f"[训练]下层{run_episodes}次训练结束")
                        self.agents_list.append((self.high_agent, self.low_agent))
                        self.last_agent = (self.high_agent, self.low_agent)
                        torch.save(self.high_agent.actor.state_dict(),
                                   f"{setting.model_path}_{CUL_FILE_NAME}_high_model_{run_episodes}.pth")
                        torch.save(self.low_agent.actor.state_dict(),
                                   f"{setting.model_path}_{CUL_FILE_NAME}_low_model_{run_episodes}.pth")

                        run_episodes += 1
                        time.sleep(setting.train_timespan)  # 训练速度太快了，评价跟不上
                    else:
                        time.sleep(1)
                self.training = False

if __name__=='__main__':
    train = TrainingPipeline()
    train.data_collector()