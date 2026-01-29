#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2025/4/11 15:35
@File:train_main.py
@Desc:训练函数
"""
# 训练函数包含三个线程实现，一个线程用于收集数据并进行处理放入经验池，一个线程从经验池中获取数据进行训练，最后一个线程
# 定时获取动作发送会控制平面
import random
import time
import torch
from utils import log_management, ReplayBuffer
import asyncio
import setting
from concurrent.futures import ThreadPoolExecutor
import requests
from data_fraction import State,get_high_action_pro,get_high_reward
import copy
import numpy as np
from algorithms_model import HighDDPG

# 训练流水线
class TrainingPipeline:
    def __init__(self):
        self.high_buffer = ReplayBuffer(setting.high_buffer_size)
        self.low_buffer = ReplayBuffer(setting.low_buffer_size)
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._running = False
        self.data_server_ip = setting.data_server_ip    # 数据服务器ip
        self.data_server_port = setting.data_server_port   # 数据服务器port

    async def data_collector(self):
        """
        数据采集线程
        :return:
        """
        last_timestamp = None
        last_high_state = None
        last_high_action = None
        last_high_reward = None
        while self._running:
            await asyncio.sleep(2)
            print('数据采集。。。')
            # 获取状态信息
            response = requests.get(f'http://{self.data_server_ip}:{self.data_server_port}/api/state')
            sample_data = response.json()
            if sample_data['timestamp'] == last_timestamp:
                continue
            else:
                last_timestamp = sample_data['timestamp']
                high_state = State(sample_data)
                high_action = get_high_action_pro(sample_data)
                high_reword = get_high_reward(sample_data, high_action)

                if last_high_state == None:
                    last_high_state = copy.deepcopy(high_state)
                    last_high_action = copy.deepcopy(high_action)
                    last_high_reward = copy.deepcopy(high_reword)
                else:
                    _high_state = copy.deepcopy(last_high_state)
                    _high_action = copy.deepcopy(last_high_action)
                    _high_reward = copy.deepcopy(last_high_reward)
                    high_experience = (_high_state, _high_action, _high_reward, high_state)
                    self.high_buffer.add(high_experience)  # 保存上层数据
                    # 保存下层数据
                    low_samples = high_state.get_low_sample()
                    for low_experience in low_samples:
                        self.low_buffer.add(low_experience)  # 保存下层数据

                    last_high_state = copy.deepcopy(high_state)
                    last_high_action = copy.deepcopy(high_action)
                    last_high_reward = copy.deepcopy(high_reword)

                if self.high_buffer.size() % 100 == 0:
                    print(f"[采样] 上层经验池大小：{self.high_buffer.size()}。")
                if self.low_buffer.size() % 100 == 0:
                    print(f"[采样] 下层经验池大小：{self.low_buffer.size()}。")


    async def training_loop(self):
        """
        训练线程
        :return:
        """
        while self._running:
            await asyncio.sleep(random.uniform(0.005, 0.01))
            print("训练中。。。")
            high_actor_lr = setting.high_actor_lr  # 上层策略网络的学习率
            high_critic_lr = setting.high_critic_lr  # 上层评价网络的学习率
            high_num_episodes = setting.high_num_episodes  # 上层迭代次数
            high_hid_dim = setting.high_hid_dim  # 上层隐藏层的维度
            high_dropout = setting.high_dropout  # 上层参数丢弃率
            high_gamma = setting.high_gamma  # 上层折扣因子
            high_tau = setting.high_tau  # 上层软更新的比率
            high_buffer_size = setting.high_buffer_size  # 上层buffer的大小
            high_buffer_req_size = setting.high_buffer_req_size  # 上层开始训练的buffer大小需求
            high_sigma = setting.high_sigma  # 上层高斯标准差
            high_nefeat = len(setting.edge_attr_choice)   # 边特征数量，输入特征数
            high_cri_nefeat = high_nefeat     # 上层评价网络输入维数
            high_act_nout = len(setting.src_dst_pair)  # 上层动作网络输出特征维数
            high_batch_size = setting.high_batch_size
            device = setting.device   # 设备
            print("训练中。。。2")
            agent = HighDDPG(high_nefeat, high_hid_dim, high_act_nout, high_cri_nefeat, high_actor_lr, high_critic_lr,
                             high_dropout, high_gamma, high_sigma, high_tau, device)

            for i_episode in range(high_num_episodes):
                sample_batch = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.high_buffer.sample(high_batch_size))


                agent.update(sample_batch)











            batch = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.high_buffer.sample(64)
            )
            # 模拟训练耗时
            await asyncio.sleep(random.uniform(0.005,0.01))
            print(f'[训练]采样大小为{len(batch)}的过程。')

    async def control_loop(self):
        """
        控制平面通信线程
        :return:
        """
        while self._running:
            await asyncio.sleep(1)
            status = {
                "high_buffer": self.high_buffer.size(),
                "low_buffer": self.low_buffer.size(),
                "timestamp": time.time()
            }
            print(f'[控制]系统状态：{status}')

    async def run(self):
        self._running = True
        # 随机种子设定
        random.seed(setting.seed)
        np.random.seed(setting.seed)
        torch.manual_seed(setting.seed)
        await asyncio.gather(
            self.data_collector(),
            self.training_loop(),
            self.control_loop()
        )

    def stop(self):
        self._running = False
        self.executor.shutdown()


if __name__ == '__main__':
    formatted_time = time.strftime("%Y-%m-%d_%H:%M", time.localtime())
    print(f'\n\n{formatted_time}运行结果\n\n')
    # logger = log_management()

    # high_buffer_size = 10000  # 上层经验池大小
    # low_buffer_size = 5000    # 下层经验池大小
    #
    # high_replay_buffer = ReplayBuffer(high_buffer_size)  # 上层经验池
    # low_replay_buffer = ReplayBuffer(low_buffer_size)  # 下层经验池
    pipeline = TrainingPipeline()

    try:
        # 启动训练流水线
        asyncio.run(pipeline.run())
    except KeyboardInterrupt:
        pipeline.stop()
        print("训练流水线下机")






    # # 异步提交数据收集过程
    # asyncio.run(add(high_replay_buffer))
    #
    # for i in range(100):
    #     time.sleep(5)
    #     print(high_replay_buffer.size())










