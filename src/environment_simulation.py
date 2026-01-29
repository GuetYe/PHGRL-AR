#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2024/12/1 21:19
@File:environment_simulation.py
@Desc:环境仿真
  - 环境仿真主要包含两个类，主机类（Host）和路由器类(Router)
  - 主机类，包括两个状态，一个是任务的产生和发送
"""
import collections
import random
from datetime import datetime,timedelta
from src.utils import EventManager
import pandas as pd
import os
import glob
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import re
import asyncio

# 导入日志管理
import sys
sys.path.append('..')  # 环境设置
from src.utils import log_management
import src.hyperparameter_tuning as ht  # 参数设置
# 1.导入日志系统 -- 作为全局变量
file_name = os.path.basename(__file__).split('.')[0]
print_level = ht.print_level
file_level = ht.file_level
logger = log_management(file_name, print_level, file_level)



class Packet:
    """ 表示一个数据包 """
    def __init__(self, source, destination, protocol, length, data_num, data, dateline_time):
        self.source = source  # 源主机
        self.destination = destination  # 任播目的主机有多个
        self.protocol = protocol  # 协议(TCP)
        self.length = length  # 数据包大小
        self.data_num = data_num  # 数据包编号
        self.data = data  # 数据内容
        self.dateline_time = dateline_time  # 数据包传输时间，超过这时间将进行重传

# 接收确认
class Host:
    """ 主机，负责生成数据并发送 """
    def __init__(self, name, location, queue = 0.0, transfer_rate=0.01):
        import src.hyperparameter_tuning as ht
        self.name = name  # 主机名字
        self.queue = queue  # 数据占用空间
        self.transfer_rate = transfer_rate  # 传输速率属性
        self.last_gen_packet_time = ht.base_time   # 上一次生成数据的时间,格式为时间戳

        # 下面两个变量用于绘图
        self.time_series = [0]  # 时间序列
        self.used_space_series = [queue]  # 数据占用序列


    # 主机生成数据包
    async def gen_packet(self, gen_timestamp, packet):  # packet:Packet对象, current_time:时间戳
        """
        :param packet:
        :param current_time: 数据生成的时间
        :return:
        """
        # 主机的数据量加上对应数据的大小
        # 主机数据量变化
        # 计算数据包发送的时间
        gen_timestamp = datetime.strptime(gen_timestamp,"%Y-%m-%d %H:%M:%S.%f").replace(microsecond=0)
        current_used_space = max(self.queue - self.transfer_rate * (gen_timestamp-self.last_gen_packet_time).total_seconds(),0)
        send_packet_time = gen_timestamp + timedelta(seconds=current_used_space/self.transfer_rate) # 时间戳的运算

        print('该代码被执行')
        await asyncio.sleep(2)
        print('停顿后执行')
        # 更新数据容量: 剩余数据量 + 到达包的大小
        self.used_space = current_used_space + packet.length
        # 记录时间和数据
        self.time_series.append(gen_timestamp)
        self.used_space_series.append(self.used_space)

        # 更新上一次数据生成的时间
        self.last_gen_packet_time = gen_timestamp

        # 生成任务
        # TODO 生成新的任务


    # 发送数据包
    def send_packet(self, dest_host, data):
        # 发送数据包 --> 接收数据包
        # TODO 输出事件日志 print(f"{self.name}:生成数据并发送到{dest_host.name}")
        packet = Packet(self.name, dest_host.name, size, number)
        return packet

    def receive_packet(self):
        # 接收确认时间
        pass

class Router:
    """ 路由节点，负责转发数据包 """
    def __init__(self, name, queue = 0, transfer_rate=0.001):
        import hyperparameter_tuning as ht
        self.name = name
        self.buffer = collections.deque(maxlen=ht.CAPACITY)
        # 排队时延，传输时延，传播时延

    # 接收数据包
    def receive_packet(self,packet):
        pass

    # 发送数据包
    def send_packet(self,packet):
        pass


import src.hyperparameter_tuning as ht
# 拓扑解析
class Topology_analysis:
    def __init__(self,xml_path):
        self.xml_path = xml_path
        self.G,self.node_dicts = self._parse_xml()  # 生成节点拓扑图和节点功能

    # 解析.xml成图
    def _parse_xml(self):
        tree = ET.parse(self.xml_path)  # 将.xml文件解析为tree
        root = tree.getroot()
        nodes_element = root.find("topology").find("nodes")
        links_element = root.find("topology").find("links")
        nodes_dicts = {}  # 节点映射，将topo节点与对象功能之间做映射
        G = nx.Graph()

        # 解析节点
        for child in nodes_element.iter():    # 遍历节点元素
            if child.tag == 'node':
                node_id = child.get('id')
                name = child.find('category').get('name')
                G.add_node(name)
                # print(node_id)
                queue_length = float(child.find('queue').get('length'))  # 队列长度初始值
                transfer_rate = float(child.find('transfer').get('rate'))  # 数据传输速率
                # 随机生成节点位置横纵坐标，(经度-90，90；维度-180,180)的范围内，主机的位置近似为节点所在的位置横纵坐标加减1。
                latitude = random.uniform(ht.lat_min, ht.lat_max)  # 经度
                longitude = random.uniform(ht.lon_min, ht.lon_max)  # 维度
                location = {'latitude': latitude, 'longitude': longitude}  # 位置坐标表示
                nodes_dicts[name] = Host(name, location, queue_length, transfer_rate)  # 建立节点名字到对象之间的映射

        # 解析链路
        for child in links_element.iter():  # 遍历链路元素
            if child.tag == 'link':
                link_id = child.get('id')
                src,std = [node for node in link_id.split('-')]
                G.add_edge(src,std)  # 将边添加到图
                for subnode in child:
                    if subnode.tag != 'id':  # 忽略'id'，因为它没有属性
                        # 获取所有属性
                        for attr_value in subnode.attrib.values():  # 遍历链路的所有属性
                            attr_value = float(attr_value)
                            G[src][std][subnode.tag] = attr_value  # 为图添加属性
                            # print(subnode.tag,attr_value)
        return G,nodes_dicts


    def draw(self):
        # pos = nx.spring_layout(self.G)  # 定义节点位置
        # nx.draw(self.G, pos, with_labels=True, node_size=500, node_color="skyblue",
        #         font_size=20, font_weight="bold")
        # plt.show()
        # 将原点显示成菱形黄色      菱形：‘d’  黄色：‘y’
        # 将终点显示成六边形红色        六边形：‘H’  红色：‘r’
        # 其他节点显示成圆形蓝色        圆形：‘o’   蓝色：‘b’
        # pos = nx.circular_layout(self.G)  # 定义节点位置
        # pos = nx.shell_layout(self.G)  # 定义节点位置
        # pos = nx.kamada_kawai_layout(self.G)  # 定义节点位置
        pos = nx.spring_layout(self.G,seed=42)  # 定义节点位置
        node_color = []
        for node in self.G.nodes:  # 将主机和路由器区分颜色
            if 'H' in node:
                node_color.append('r')
            if 'R' in node:
                node_color.append('b')
        nx.draw(self.G, pos,with_labels=True, node_size=500,
                node_color=node_color, font_size=10, font_weight="bold")
        edge_labels = nx.get_edge_attributes(self.G,'weight')  # 获取权重
        nx.draw_networkx_edge_labels(self.G,pos,edge_labels=edge_labels)  # 将权重绘制到图中
        topo_name = re.split(r'[\./]',self.xml_path)[-2]
        # print(re.split(r'[\./]',self.xml_path)[-2])
        plt.savefig(f'../data/figure/{topo_name}_{datetime.now().strftime("%Y-%m-%d")}.png')
        plt.show()


def main():
    # 1.确保异步函数可以运行
    # print('异步函数')
    # 2.导入拓扑
    xml_path = ht.xml_path
    topo = Topology_analysis(xml_path)
    # 3.测试通过拓扑可以调用对象
    src_set = ht.src_set  # 源节点集合
    std_set = ht.std_set  # 目的节点集合
    # # 遍历源节点，访问它对象
    # for src in src_set:
    #     print(topo.node_dicts[src].name)  # 测试通过
    # # 4.测试通过交换节节点可以调用对象
    # print(topo.node_dicts['R1'].name)  # 测试通过
    # 生成文件名
    data_path = '../data/database'
    data_root = '_data_arrival'
    task_list = []  # 事件列表
    # 5.遍历源节点集，生成数据
    for src in src_set:
        prefix = src+data_root                # 文件名前缀
        pattern = os.path.join(data_path,f'{prefix}*')
        match_file_name = glob.glob(pattern)   # 部分匹配
        assert len(match_file_name) == 1, f'文件名前缀{prefix}匹配对象个数为{len(match_file_name)}，不唯一，请检查。'
        packet_df = pd.read_csv(match_file_name[0])  # 解压文件
        print(packet_df)
        for index,row in packet_df.iterrows():
            # 构建数据包
            source = src  # 数据包源
            destination = std_set  # 数据包目的地
            protocol = ht.protocol  # 数据包协议
            length = row['Packet_size(bit)']  # 数据包大小
            data_num = index  # 数据包编号
            data = ht.data  # 数据包内容
            dateline_time = ht.dateline_time  # 数据包确认时间
            packet = Packet(source,destination,protocol,length,data_num,data,dateline_time)  # 生数据包
            # 异步数据包生成事件
            # print('速率',topo.node_dicts[src].transfer_rate)
            task_list.append(topo.node_dicts[src].gen_packet(row['Arrival_Timestamp'],packet))
            print("---进入事件前---")
            asyncio.run(asyncio.wait(task_list))
            print("---进入事件后---")
















if __name__ == '__main__':
    main()



