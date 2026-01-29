#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2025/4/15 20:52
@File:train_main_thread.py
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
        self.ctrl_queue = queue.Queue(maxsize=100)
        self.data_queue = queue.Queue(maxsize=1000)

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
        self.last_agent = None # 最新智能体，训练期间用

        self.training = True  # 模型训练中的标识
        self.evaluating = True  # 模型评价中的标识

    def data_collector(self):
        """数据采集线程（I/O密集型）"""
        last_timestamp = None
        last_high_state = None
        while self._running:
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

            loss_test_list = [e['loss']<1 for e in sample_data['edges']]  # 检查loss>1的数据异常
            if not all(loss_test_list):
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

    def training_loop(self):
        """训练线程（GPU密集型）"""
        # 初始化训练环境
        high_train_model = setting.high_train_model
        if high_train_model == 'DDPG':
            high_nfeat_e = len(setting.edge_attr_choice)  # 边特征数量，输入特征数
            high_nhid = setting.high_hid_dim  # 上层隐藏层的维度
            high_nout = len(setting.src_dst_pair)  # 上层动作网络输出特征维数
            high_act_lr = setting.high_actor_lr  # 上层策略网络的学习率
            high_cri_lr = setting.high_critic_lr  # 上层评价网络的学习率
            high_dropout = setting.high_dropout  # 上层参数丢弃率
            high_gamma = setting.high_gamma  # 上层折扣因子
            high_sigma = setting.high_sigma  # 上层高斯标准差
            high_tau = setting.high_tau  # 上层软更新的比率
            device = setting.device  # 设备

            self.high_agent = HighDDPG(
                high_nfeat_e, high_nhid, high_nout,
                high_act_lr, high_cri_lr, high_dropout, high_gamma,
                high_sigma, high_tau, device
            )

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
            while self._running:
                if self.high_buffer.size() >= setting.high_buffer_req_size and self.low_buffer.size() >= setting.low_buffer_req_size:
                    high_sample_batch = self.high_buffer.sample(setting.high_batch_size)
                    if high_sample_batch:
                        self.high_agent.update(high_sample_batch)
                        print(f"[训练]上层{run_episodes}次训练结束")
                    low_sample_batch = self.low_buffer.sample(setting.low_batch_size)
                    if low_sample_batch:
                        self.low_agent.update(low_sample_batch)
                        print(f"[训练]下层{run_episodes}次训练结束")
                    torch.save(self.high_agent.actor.state_dict(),f"{setting.model_path}_{CUL_FILE_NAME}_high_model_{run_episodes}.pth")
                    torch.save(self.low_agent.actor.state_dict(),f"{setting.model_path}_{CUL_FILE_NAME}_low_model_{run_episodes}.pth")

                    run_episodes += 1
                    time.sleep(setting.train_timespan)  # 训练速度太快了，评价跟不上
                else:
                    time.sleep(1)

        elif high_train_model == 'PPO':
            high_nfeat_e = len(setting.edge_attr_choice)  # 边特征数量，输入特征数
            high_nhid = setting.high_hid_dim  # 上层隐藏层的维度
            high_nout = len(setting.src_dst_pair)  # 上层动作网络输出特征维数
            high_act_lr = setting.high_actor_lr  # 上层策略网络的学习率
            high_cri_lr = setting.high_critic_lr  # 上层评价网络的学习率
            high_dropout = setting.high_dropout  # 上层参数丢弃率
            high_gamma = setting.high_gamma  # 上层折扣因子
            device = setting.device  # 设备
            high_lmbda = setting.lmbda
            high_epochs = setting.epochs
            high_eps = setting.eps
            self.high_agent = HighPPO(
                high_nfeat_e, high_nhid, high_nout, high_dropout,
                high_act_lr, high_cri_lr,
                high_lmbda, high_epochs, high_eps,
                high_gamma, device
            )
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

        elif high_train_model == 'SAC':
            high_nfeat_e = len(setting.edge_attr_choice)  # 边特征数量，输入特征数
            high_nhid = setting.high_hid_dim  # 上层隐藏层的维度
            high_nout = len(setting.src_dst_pair)  # 上层动作网络输出特征维数
            high_act_lr = setting.high_actor_lr  # 上层策略网络的学习率
            high_cri_lr = setting.high_critic_lr  # 上层评价网络的学习率
            high_alpha_lr = setting.high_alpha_lr   # 上层系数alpha的学习率
            high_dropout = setting.high_dropout  # 上层参数丢弃率
            high_gamma = setting.high_gamma  # 上层折扣因子
            high_tau = setting.high_tau
            high_target_entropy = -len(setting.src_dst_pair)
            device = setting.device  # 设备

            self.high_agent = HighSAC(high_nfeat_e,high_nhid,high_nout,high_act_lr,high_cri_lr,high_dropout,high_alpha_lr
                                      ,high_target_entropy,high_gamma,high_tau,device)
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




    def evaluate_loop(self):
        """评价线程"""
        # 拷贝模型
        # last_episode_count = None
        # eva_episode_list = []   # 用于存储模型对应的训练代数
        optional_reward = -float('inf')
        eva_high_reward_list = []
        eva_low_reward_list = []
        eva_count_list = []  # 用于列出评价的代数
        eva_count = 0
        eva_num_sample = setting.high_buffer_req_size if setting.high_buffer_req_size < 10 else 10

        while self._running:
            # with torch.no_grad():
            print(f'模型长度：{len(self.agents_list)}')
            if len(self.agents_list)>0:
                eva_count += 1
                high_agent,low_agent = self.agents_list.pop(0)
                high_reward_ave, low_reward_dict_ave = self.evaluate(high_agent, low_agent, num_sample=eva_num_sample)
                # eva_episode_list.append(high_episode_count)
                print(f'上层奖励均值：{high_reward_ave}')
                if high_reward_ave>optional_reward:
                    self.optional_agent = (high_agent,low_agent)

                eva_high_reward_list.append(high_reward_ave)
                eva_low_reward_list.append(low_reward_dict_ave)
                eva_count_list.append(eva_count)

                # 记录评价过的代数
                # last_episode_count = high_episode_count
                # print(f'第{eva_count}次进行评价。上层平均奖励为{eva_high_reward_list},下层平均奖励为{eva_low_reward_list}')

                # 异步提交评价绘图，保存数据
                if eva_count > 0 and eva_count % setting.eva_display_interval == 0:
                    # loop = asyncio.new_event_loop()
                    # loop.run_until_complete(self.evaluate_tool.data_display(eva_count_list=eva_count_list,
                    #                                                         eva_value_list=eva_high_reward_list,
                    #                                                         tag_name=f'{CUL_FILE_NAME}_{eva_count}',
                    #                                                         disp_mode='H'))
                    self.evaluate_tool.data_display(eva_count_list=eva_count_list,
                                                    eva_value_list=eva_high_reward_list,
                                                    tag_name=f'{CUL_FILE_NAME}_{eva_count}',
                                                    disp_mode='H')
                    # loop = asyncio.new_event_loop()
                    # loop.run_until_complete(self.evaluate_tool.data_display(eva_count_list=eva_count_list,
                    #                                                         eva_value_list=eva_low_reward_list,
                    #                                                         tag_name=f'{CUL_FILE_NAME}_{eva_count}',
                    #                                                         disp_mode='L'))
                    self.evaluate_tool.data_display(eva_count_list=eva_count_list,
                                                    eva_value_list=eva_low_reward_list,
                                                    tag_name=f'{CUL_FILE_NAME}_{eva_count}',
                                                    disp_mode='L')
                print(f'[评价]第{eva_count}次评价结束')
                time.sleep(setting.eva_timespan)

            elif self.training:
                print(f'[评价]还没有待评估的模型，请稍等。。。')
                time.sleep(1)
                continue

            else:
                break
        self.evaluating = False


                # high_agent = copy.deepcopy(self.high_agent)
                # low_agent = copy.deepcopy(self.low_agent)
                # if high_agent==None or low_agent==None:
                #     print(f'[评价]模型还没准备好，请重试。。。')
                #     time.sleep(1)
                #     continue
                # else:
                #
                #     # 读取最新的模型参数
                #     high_agent_last_file_name = read_last_weight_file(setting.model_path, identifier="high")
                #     # low_agent_last_file_name = read_last_weight_file(setting.model_path, identifier="low")
                #     if high_agent_last_file_name:
                #         low_agent_last_file_name = high_agent_last_file_name.replace('high','low')
                #         # 提取训练代数
                #         high_episode_count = int(re.split(r'[\._]', high_agent_last_file_name)[-2])
                #         low_episode_count = int(re.split(r'[\._]', low_agent_last_file_name)[-2])
                #         if high_episode_count != low_episode_count:
                #             print(f'[评价]上层{high_episode_count}下层{low_episode_count}模型的迭代次数不统一，请检查。。。')
                #             time.sleep(1)
                #             continue
                #         elif high_episode_count == last_episode_count:
                #             print(f'[评价]第{high_episode_count}代模型已经评价过，无需重复评价。')
                #             time.sleep(1)
                #             continue
                #         else:
                #             eva_count += 1
                #
                #             # 读取模型参数
                #             high_agent_parameters = torch.load(high_agent_last_file_name)
                #             low_agent_parameters = torch.load(low_agent_last_file_name)
                #             # 模型参数赋值
                #             high_agent.actor.load_state_dict(high_agent_parameters)
                #             low_agent.actor.load_state_dict(low_agent_parameters)
                #             # 进入评价模式
                #             high_reward_ave,low_reward_dict_ave = self.evaluate(high_agent,low_agent,num_sample=2)
                #             eva_episode_list.append(high_episode_count)
                #             eva_high_reward_list.append(high_reward_ave)
                #             eva_low_reward_list.append(low_reward_dict_ave)
                #             eva_count_list.append(eva_count)
                #
                #             # 记录评价过的代数
                #             last_episode_count = high_episode_count
                #             # print(f'第{eva_count}次进行评价。上层平均奖励为{eva_high_reward_list},下层平均奖励为{eva_low_reward_list}')
                #
                #             # 异步提交评价绘图，保存数据
                #             if eva_count>0 and eva_count % setting.eva_display_interval == 0:
                #                 loop = asyncio.new_event_loop()
                #                 loop.run_until_complete(self.evaluate_tool.data_display(eva_count_list=eva_count_list,
                #                                 eva_value_list=eva_high_reward_list,eva_count_range=eva_episode_list,
                #                                 tag_name=f'{CUL_FILE_NAME}_{eva_count}'))
                #                 loop = asyncio.new_event_loop()
                #                 loop.run_until_complete(self.evaluate_tool.data_display(eva_count_list=eva_count_list,
                #                                 eva_value_list=eva_low_reward_list,eva_count_range=eva_episode_list,
                #                                 tag_name=f'{CUL_FILE_NAME}_{eva_count}'))
                #             print(f'[评价]第{eva_count}次评价结束')
                #
                #     else:
                #         print(f'[评价]还没有待评估的模型，请稍等。。。')
                #         time.sleep(1)
                #         continue



    def evaluate(self, high_agent, low_agent, num_sample=10):
        """
        评价方法
        :param high_agent:上层模型
        :param low_agent: 下层模型
        :param num_sample: 样本数量
        :return:
        """
        # 上层采样
        low_agent = copy.deepcopy(low_agent)
        # with torch.no_grad():
        if self.high_buffer.size() >= num_sample:
            high_sample_batch = self.high_buffer.sample(num_sample)
            if high_sample_batch:
                state_list = []
                act_list = []
                rew_list = []
                next_state_list = []
                for sample in high_sample_batch:
                    state, act, rew, next_state = sample[0]
                    state_list.append(state)
                    act_list.append(act)
                    rew_list.append(rew)
                    next_state_list.append(next_state)

                _act_input = gen_high_act_input_batch(state_list).to(setting.device)  # 当前状态
                with torch.no_grad():
                    if setting.high_train_model == 'SAC':
                        _act, _ = high_agent.actor(_act_input)
                    else:
                        _act = high_agent.actor(_act_input)

                # 计算平均奖励
                high_reward_list = []
                low_reward_dict = {}
                for key in setting.src_dst_pair:
                    low_reward_dict.setdefault(key, [])
                low_reward_dict_ave = copy.deepcopy(low_reward_dict)

                # print(f'奖励输出形式{_act}')
                for state,act in zip(state_list, _act):
                    act_pro = act.tolist()  # 上层动作
                    if setting.high_train_model == 'PPO':
                        act_key = weighted_random(act_pro)
                        act_pro = list(setting.choice_pro.keys()).index(act_key)

                    # 下层评价
                    new_state_data, low_path_dict, _low_reward_dict = take_low_agent_act(low_agent, state,evaluate=False)
                    # print(f'评价中下层路径字典{low_path_dict}，奖励字典{_low_reward_dict}')
                    # 上层评价
                    high_reward = get_high_reward(new_state_data, act_pro)

                    for key in setting.src_dst_pair:
                        low_reward_dict[key].append(_low_reward_dict[key])
                    high_reward_list.append(high_reward)
                    if high_reward > 0:
                        LOGGER.info(f'计算数据：{new_state_data}，动作概率：{act_pro}')


            # print(f'平均前的数据：{high_reward_list}')
            high_reward_ave = np.mean(high_reward_list)
            for key in setting.src_dst_pair:
                low_reward_dict_ave[key] = np.mean(low_reward_dict[key])

        else:
            print(f"样本空间不足，请稍等。。。")
            time.sleep(1)
            high_reward_ave = None
            low_reward_dict_ave = None

        if high_reward_ave > 0:
            LOGGER.info(f'奖励数据：{high_reward_list}，平均奖励：{high_reward_ave}')

        return high_reward_ave, low_reward_dict_ave


    def control_loop(self):
        """控制线程"""
        # 1. 观测
        last_timestamp = None
        last_high_state = None
        last_episode_count = None
        control_count = 0
        optional_control = 0
        last_control_high_action = None
        last_control_low_action = None
        while self._running:
            # try:
            response = requests.get(
                f'http://{self.data_server_ip}:{self.data_server_port}/api/request',
                timeout=5
            )
            sample_data = response.json()

            if 'timestamp' not in sample_data.keys():
                print(f'数据{sample_data}时间搓缺失，重新获取数据！')
                continue
            elif sample_data['timestamp'] == last_timestamp:
                print(f'等待状态请求的更新，请稍后。。。')
                time.sleep(1)
                continue

            src_dst_pair = setting.src_dst_pair
            cond_list = []
            for key in src_dst_pair:
                cond_list.append(1 if key in sample_data['edges'][0].keys() else 0)
            for key in setting.edge_attr_choice:
                cond_list.append(1 if key in sample_data['edges'][0].keys() else 0)
            if not all(cond_list):
                print("路径信息没收集全，请等待。。。")
                continue

            loss_test_list = [e['loss'] < 1 for e in sample_data['edges']]  # 检查loss>1的数据异常
            if not all(loss_test_list):
                continue

            # with torch.no_grad():
            # 处理数据采集逻辑
            high_state = State(sample_data)

            # 2.决策
            # 获取actor网络的输入
            high_state_input = gen_high_act_input_batch([high_state]).to(setting.device)

            # 选择模型
            action_dict = {}  # 用于交互的动作字典
            if setting.newest_model:
                if self.training and self.last_agent:
                    action_dict['control_tag'] = 'T'
                    high_agent,low_agent = self.last_agent
                    with torch.no_grad():
                        if setting.high_train_model == 'SAC':
                            high_act, _ = high_agent.actor(high_state_input)
                        else:
                            high_act = high_agent.actor(high_state_input)
                        # high_act = high_agent.actor(high_state_input)
                    high_act_list = high_act.tolist()
                    if all(math.isnan(act) for act in high_act_list[0]):
                        action_dict['high_act_list'] = last_control_high_action
                        print(f'[控制]上层动作{high_act_list}出现{None},使用动作重现控制：{last_control_high_action}')
                        high_act_list = last_control_high_action
                        action_dict['low_act_dict'] = last_control_low_action
                        low_path_dict = last_control_low_action
                    else:
                        action_dict['high_act_list'] = high_act_list
                        print(f'[控制]上层动作：{high_act_list}')
                        _, low_path_dict, _ = take_low_agent_act(low_agent, high_state)
                        action_dict['low_act_dict'] = low_path_dict

                    # 上传控制动作
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self.data_sender.send_data(action_dict, setting.data_server_action_path))
                    # self.data_sender.send_data(action_dict, setting.data_server_action_path)
                    time.sleep(setting.control_timespan)
                    control_count += 1
                    print(f'[控制]第{control_count}次最新模型控制结束')
                    status = {
                        "high_buffer": self.high_buffer.size(),
                        "low_buffer": self.low_buffer.size(),
                        "timestamp": time.time()
                    }
                    print(f'[控制]系统状态：{status}')


                elif not self.training and self.optional_agent:
                    action_dict['control_tag'] = 'C'
                    high_agent, low_agent = self.optional_agent
                    with torch.no_grad():
                        if setting.high_train_model == 'SAC':
                            high_act, _ = high_agent.actor(high_state_input)
                        else:
                            high_act = high_agent.actor(high_state_input)
                        # high_act = high_agent.actor(high_state_input)
                    high_act_list = high_act.tolist()
                    action_dict['high_act_list'] = high_act_list
                    print(f'[控制]上层动作：{high_act_list}')
                    _, low_path_dict, _ = take_low_agent_act(low_agent, high_state)
                    action_dict['low_act_dict'] = low_path_dict

                    # 上传控制动作
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(
                        self.data_sender.send_data(action_dict, setting.data_server_action_path))
                    # self.data_sender.send_data(action_dict, setting.data_server_action_path)
                    time.sleep(setting.control_timespan)
                    control_count += 1
                    optional_control += 1
                    print(f'[控制]第{control_count}次最优控制结束')
                    status = {
                        "high_buffer": self.high_buffer.size(),
                        "low_buffer": self.low_buffer.size(),
                        "timestamp": time.time()
                    }
                    print(f'[控制]系统状态：{status}')

                else:
                    print(f'[控制]还没有待评估的模型，请稍等。。。')
                    status = {
                        "high_buffer": self.high_buffer.size(),
                        "low_buffer": self.low_buffer.size(),
                        "timestamp": time.time()
                    }
                    print(f'[控制]系统状态：{status}')
                    time.sleep(1)
                    continue
                last_control_high_action = high_act_list
                last_control_low_action = low_path_dict
            # if optional_control == setting.optional_control:
            #     self._running = 0  # 线程停止

                # high_agent = copy.deepcopy(self.high_agent)
                # low_agent = copy.deepcopy(self.low_agent)
                # if high_agent == None or low_agent == None:
                #     print(f'[Evaluate]模型还没准备好，请重试。。。')
                #     time.sleep(1)
                #     continue
                # else:
                #     # 读取最新的模型参数
                #     high_agent_last_file_name = read_last_weight_file(setting.model_path, identifier="high")
                #     # low_agent_last_file_name = read_last_weight_file(setting.model_path, identifier="low")
                #     if high_agent_last_file_name:
                #         low_agent_last_file_name = high_agent_last_file_name.replace('high', 'low')
                #         # 提取训练代数
                #         high_episode_count = int(re.split(r'[\._]', high_agent_last_file_name)[-2])
                #         low_episode_count = int(re.split(r'[\._]', low_agent_last_file_name)[-2])
                #         if high_episode_count != low_episode_count:
                #             print(f'[Evaluate]上层{high_episode_count}下层{low_episode_count}模型的迭代次数不统一，请检查。。。')
                #             time.sleep(1)
                #             continue
                #         elif high_episode_count == last_episode_count:
                #             print(f'[Evaluate]第{high_episode_count}代模型已经评价过，无需重复评价。')
                #             time.sleep(1)
                #             continue
                #         else:
                #             control_count += 1
                #             action_dict['state_timestamp'] = sample_data['timestamp']
                #             last_timestamp = sample_data['timestamp']
                #             # 读取模型参数
                #             high_agent_parameters = torch.load(high_agent_last_file_name)
                #             low_agent_parameters = torch.load(low_agent_last_file_name)
                #             # 模型参数赋值
                #             high_agent.actor.load_state_dict(high_agent_parameters)
                #             low_agent.actor.load_state_dict(low_agent_parameters)
                #             # 进入控制模式
                #
                #
                #             high_act = high_agent.actor(high_state_input)
                #             high_act_list = high_act.tolist()
                #             action_dict['high_act_list'] = high_act_list
                #             print(f'[控制]上层动作：{high_act_list}')
                #             _, low_path_dict, _ = take_low_agent_act(low_agent, high_state)
                #             action_dict['low_act_dict'] = low_path_dict
                #
                #             # 上传控制动作
                #             loop = asyncio.new_event_loop()
                #             loop.run_until_complete(self.data_sender.send_data(action_dict, setting.data_server_action_path))
                #             time.sleep(setting.control_timespan)
                #             print(f'[控制]第{control_count}次控制结束')
                #
                #     else:
                #         print(f'[Control]还没有待评估的模型，请稍等。。。')
                #         time.sleep(1)
                #         continue


    def run(self):
        """启动线程系统"""
        random.seed(setting.seed)
        np.random.seed(setting.seed)
        torch.manual_seed(setting.seed)
        threads = []

        # 创建数据采集线程
        for _ in range(setting.num_collectors):
            t = threading.Thread(target=self.data_collector, daemon=True)
            threads.append(t)

        # 创建训练线程
        for _ in range(setting.num_learners):
            t = threading.Thread(target=self.training_loop, daemon=True)
            threads.append(t)

        # 创建评价循环
        t = threading.Thread(target=self.evaluate_loop, daemon=True)
        threads.append(t)

        # 创建控制线程
        t = threading.Thread(target=self.control_loop, daemon=True)
        threads.append(t)

        # 启动所有线程
        for t in threads:
            t.start()

        # 主线程监控
        try:
            while True:
                if not self.ctrl_queue.empty():
                    status = self.ctrl_queue.get_nowait()
                    # print(f"[系统状态] 上层缓存：{status['high_buffer']}，下层缓存：{status['low_buffer']}")
                time.sleep(5)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """停止系统"""
        self._running = False
        print("正在关闭系统...")
        time.sleep(1)  # 等待线程退出


if __name__ == '__main__':
    if len(sys.argv)>1:
        delele_history_data_tag = sys.argv[1]
        if delele_history_data_tag == 'c' or delele_history_data_tag == 'C':
            delete_data()
    tag = f''
    exp_init(tag)

    pipeline = TrainingPipeline()
    pipeline.run()

# 在运行前请确定是否删除历史数据

# 运行代码
# nohup python3 -u train_main_thread.py >> ../data/result/log/main_v14-1_output.log 2>&1 &

# 查询代码
# cat ../data/result/log/main_v14-1_output.log

# 杀死
# kill -9 进程号  # 26063