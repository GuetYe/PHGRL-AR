#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/12/8 10:54
@File:host_data_gen.py
@Desc:主机数据生成
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import hyperparameter_tuning as ht
from datetime import datetime,timedelta

# 设置随机种子以确保可重复性
np.random.seed(ht.seed)

def host_data_gen(host_name,num_packets,lambda_arrival,mean_size,std_size):
    # # 模拟参数
    # num_packets = ht.num_packets  # 模拟数据包的数量
    # lambda_arrival = ht.lambda_arrival  # 数据包到达的平均速率（λ参数）
    # mean_size = ht.mean_size  # 数据包大小的均值(bit)
    # std_size = ht.std_size  # 数据包大小的标准偏差(bit)

    # 生成到达时间(服从指数分布) 单位：s
    arrival_times1 = np.random.exponential(scale=1/lambda_arrival, size=num_packets)
    arrival_times = np.cumsum(arrival_times1)  # 计算累计到达时间
    # arrival_times = ht.base_time * np.ones(shape=np.shape(arrival_times)) + timedelta(arrival_times)  # 将时间转化为

    arrival_datetimes = []
    for index,num in enumerate(arrival_times):
        arrival_datetimes.append(ht.base_time + timedelta(seconds=num))
    # print(arrival_datetimes)

    # 生成数据包大小
    packet_sizes = np.random.normal(loc=mean_size, scale=std_size, size=num_packets)

    # # 打印生成的数据包信息
    # for i in range(10):  # 打印前10个数据包的到达时间和大小
    #     print(f"数据包 {i+1}: 到达时间 = {arrival_datetimes[i]}, 大小 = {packet_sizes[i]:.2f}bit")

    # 数据包到达时间分布可视化
    plt.figure(figsize=(12, 6))

    # 到达时间直方图
    plt.subplot(1, 2, 1)
    plt.hist(arrival_times1, bins=30, color='blue', alpha=0.7)
    plt.title('Packet Arrival Time Distribution (Exponential Distribution)')  # 数据包到达时间分布（指数分布）
    plt.xlabel('Arrival Time (sec)')  # 到达时间（秒）
    plt.ylabel('Frequency')  # 频率

    # 数据包大小分布可视化
    plt.subplot(1, 2, 2)
    plt.hist(packet_sizes, bins=30, color='orange', alpha=0.7)
    plt.title('Packet Size Distribution (Normal Distribution)')  # 数据包大小分布（正态分布）
    plt.xlabel('Packet Size (bit)')  # 数据包大小（字节）
    plt.ylabel('Frequency')  # 频率

    plt.tight_layout()
    # 保存图片
    plt.savefig(f'../data/figure/{host_name}_data_arrival_{datetime.now().strftime("%Y-%m-%d")}.png',dpi=300,bbox_inches='tight')
    plt.show()
    # 保存数据
    data_arrival_df = pd.DataFrame(
        {'Arrival_Time(s)': arrival_times,
         'Arrival_Timestamp': arrival_datetimes,
         'Packet_size(bit)': packet_sizes
        }
    )
    data_arrival_df.to_csv(f'../data/database/{host_name}_data_arrival_{datetime.now().strftime("%Y-%m-%d")}.csv')



# 为四个主机生成数据
if __name__ == '__main__':
    hosts_name = ht.src_set
    hosts_num_packets = ht.hosts_num_packets
    hosts_lambda_arrival = ht.hosts_lambda_arrival
    hosts_mean_size = ht.hosts_mean_size
    hosts_std_size = ht.hosts_std_size

    for index,host_name in enumerate(hosts_name):
        host_data_gen(host_name, hosts_num_packets[index], hosts_lambda_arrival[index],
                      hosts_mean_size[index], hosts_std_size[index])





