#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2025/2/25 11:47
@File:utils.py
@Desc:****************
"""
import json
# 自定义JSON序列化处理
import os.path
import traceback
import aiohttp
import logging
import time
import os

# 日志管理系统
# def log_management(file_name,print_level=logging.INFO,file_level=logging.DEBUG):
#     """
#     输出日志
#     :param file_name: 调用日志的文件名
#     :param print_level: 打印等级
#     :param file_level: 文件输出等级
#     :return:
#     """
#     formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime())
#     log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S %p")
#     # 创建logger
#     logger = logging.getLogger('run_logger')  # 创建打印日志--运行时打印的日志
#     logger.setLevel(logging.DEBUG)  # 选择打印日志的等级
#     """
#     DEBUG < INFO < WARNING < ERROR < CRITICAL
#     DEBUG: 用于测试打印(测试信息-不要紧的消息、所有信息)
#     INFO：相对有用的提示(用于操作提示)
#     WARNING：必要的信息(用于结果显示)
#     ERROR：更高级的设置
#     CRITICAL：更高级的设置
#     """
#     # 创建文件日志 file_logger
#     file_logger_handler = logging.FileHandler(f'./logging/{file_name}_run_logger_{formatted_time}.log')
#     file_logger_handler.setLevel(file_level)
#
#     # 为文件日志设置格式
#     file_logger_handler.setFormatter(log_format)
#
#     # 添加文件日志
#     logger.addHandler(file_logger_handler)
#
#     # 添加打印日志
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(print_level)
#     # 设置打印格式
#     console_handler.setFormatter(log_format)
#
#     # 添加打印日志
#     logger.addHandler(console_handler)
#
#     # 添加第一条日志
#     logger.critical(f'[{file_name}.py]_run_logger_{formatted_time}')  # 以最高级别打印标题
#
#     return logger
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
    log_dir = './logging'
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


file_name = os.path.basename(__file__).split('.')[0]
logger = log_management(file_name)

def json_default(obj):
    """处理字典中非字符串键的序列化"""
    if isinstance(obj, dict):
        return {str(k): v for k,v in obj.items()}
    raise TypeError(f"Cannot serialize {type(obj)}")


# 通过线程实时保存字典数据
def thread_save_dict(file_path,data):
    data = json_default(data)
    with open(file_path,'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


import socket
import struct
MAGIC_NUMBER = 0xDEADBEEF
from crc32c import crc32c
import msgpack
import msgpack_numpy
import networkx as nx
import setting
import json
import copy
from multiprocessing import Process

class SocketClient:
    def __init__(self,server_ip,port):
        """
        连接服务器，需要ip和端口
        :param server_ip:服务器ip
        :param port:端口
        """
        # 通过socket建立连接
        self.sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.sock.connect((server_ip,port))
        print(f'与服务器{server_ip}:{port}建立连接')
        self.seq_num = 0  # 包序号计数器
        self.src_node_list = setting.src_node_list  # 源节点列表[1]
        self.dst_node_list = setting.dst_node_list  # 目的节点列表[15,16,17,18]

    def send_graph_path_state_process(self, graph_data, path_file):
        """
        通过多线程的形式发送数据给服务器
        :param graph_data:
        :param path_file:
        :return:
        """
        send_state_process = Process(target=self.send_graph_path_state,args=(graph_data,path_file))
        send_state_process.start()



    def send_graph_path_state(self, graph_data, path_file):
        """
        在图属性的基础上添加路径的标识
        :param graph_data:
        :param path_file:
        :return:
        """
        with open(path_file, 'r', encoding='utf-8') as f:
            path_dict = json.load(f)

        new_graph_data = copy.deepcopy(graph_data)  # 对数据进行深拷贝，方便进行修改

        # 遍历边的信息
        for src in self.src_node_list:
            for dst in self.dst_node_list:
                key_name = f'({src},{dst})'  # 键名
                _path = path_dict[key_name]   # 获取路径
                _path_set = [(_src,_dst) for _src, _dst in zip(_path[:-1], _path[1:])]
                for u, v in graph_data.edges():
                    if (u, v) in _path_set:
                        new_graph_data[u, v][key_name] = 1
                    else:
                        new_graph_data[u, v][key_name] = 0


        self.send_graph(new_graph_data)


    def send_graph(self,graph_data):
        """
        发送图数据
        :param graph_data: 原图数据
        :return:
        0        4        8        12       16
        +--------+--------+--------+--------+
        |  MAGIC | SEQ_ID | LENGTH |  CRC32 |
        +--------+--------+--------+--------+
        """
        serialize_graph_data = self.serialize_graph(graph_data)
        header = struct.pack('>IIII',MAGIC_NUMBER,self.seq_num,len(serialize_graph_data),crc32c(serialize_graph_data))
        self.sock.sendall(header + serialize_graph_data)
        print('数据传输结束')
        self.seq_num += 1
        #  物理机                          服务器
        # 原图数据 ---> 序列化二进制 ---> 二进制图数据 ---> 反序列化

    def recv_loop(self):
        """
        接收结果
        :return:
        """
        while True:
            try:
                header = self.sock.recv(16)
                assert len(header) >= 16, '数据长度异常！'
                magic, seq, data_len, crc = struct.unpack('>IIII',header)

                data = b''

                while len(data) < data_len:
                    chunk = self.sock.recv(data_len-len(data))   # 解析剩余数据
                    if not chunk:
                        break
                    data += chunk

                if crc32c(data) == crc:
                    print(f"收到的数据结果为：{data}")
            except ConnectionAbortedError:
                break


    def serialize_graph(self,graph_data):
        """
        图数据序列化：原图数据--->二进制
        :param graph_data:
        :return:
        """
        graph_packed = msgpack.packb(self.networkx_to_dict(graph_data), use_bin_type=True)
        return graph_packed

    def deserialize_graph(self,serialize_graph_data):
        """
        图数据反序列化：二进制--->图数据
        :param serialize_graph_data:
        :return:
        """
        data = msgpack.unpackb(serialize_graph_data,raw=False)
        graph = nx.Graph()
        for n in data['nodes']:
            graph.add_node(n['id'],**{k:v for k,v in n.itrms() if k != 'id'})  # 反序列节点属性
        for e in data['edges']:
            graph.add_edge(e['source'],e['target'], **{k:v for k,v in e.items() if k not in ['source','target']})  # 反序列边属性
        return graph

    def networkx_to_dict(self,graph_data):
        """
        将图数据转化为字典
        :return:
        """
        return {
            "nodes": [{"id":n[0],**n[1]} for n in graph_data.nodes(data=True)],
            "edges": [{'source':u, "target":v, **d} for u,v,d in graph_data.edges(data=True)]
        }

    # 创建带权重的有向图
    def create_test_graph(self):
        # 初始化空图
        G = nx.DiGraph()

        # 添加带属性的节点
        nodes = [
            (1, {"color": "red", "size": 100}),
            (2, {"color": "blue", "size": 200}),
            (3, {"color": "green", "size": 150}),
            (4, {"color": "yellow", "size": 80}),
            (5, {"color": "purple", "size": 120}),
        ]
        G.add_nodes_from(nodes)

        # 添加带权重的边
        edges = [
            (1, 2, {"weight": 1.5}),
            (1, 3, {"weight": 2.0}),
            (2, 4, {"weight": 0.8}),
            (3, 4, {"weight": 1.2}),
            (4, 5, {"weight": 2.5}),
            (5, 1, {"weight": 0.5}),
        ]
        G.add_edges_from(edges)
        return G

    def performance_test(self):
        graph_data = self.create_test_graph()
        self.send_graph(graph_data)


from flask import Flask, jsonify
import threading
import time
import requests

# 1. 定义发送类
class SendGraphData:
    def __init__(self,server_ip, server_port,state_path,request_path,action_path):
        self.server_ip = server_ip    # 服务端ip
        self.server_port = server_port   # 服务端端口
        self.state_path = state_path   # 服务端状态路径
        self.request_path = request_path   # 服务器请求路径
        self.action_path = action_path   # 服务端动作路径
        self.src_node_list = setting.src_node_list  # 源节点列表[1]
        self.dst_node_list = setting.dst_node_list  # 目的节点列表[15,16,17,18]
        self.path_file = setting.path_file_path  # 路径文件位置
        self.send_graph_num = 0

    async def send_graph_path_state(self, graph_data, path_file):
        """
        在图属性的基础上添加路径的标识
        :param graph_data:
        :param path_file:
        :return:
        """
        if os.path.exists(path_file):
            with open(path_file, 'r', encoding='utf-8') as f:
                path_dict = json.load(f)
        else:
            path_dict = None

        new_graph_data = copy.deepcopy(graph_data)  # 对数据进行深拷贝，方便进行修改

        # 遍历边的信息
        """
        思路：将路径赋值成图的数据性
        1. 找到目标的路径：（1,15），（1,16），（1,17），（1,18）
        """
        # 优化1. 预生成所有需要标记的路径对
        path_pairs = []
        for src in self.src_node_list:
            for dst in self.dst_node_list:
                try:
                    # 统一键名格式（去除空格）
                    key = f'({src}, {dst})'
                    path = path_dict[key]
                    path_pairs.append((key,path))
                except Exception as e:
                    print(f'[WARN] 缺失路径配置：({src}, {dst}),错误：{e}')



        all_edges = list(graph_data.edges())
        # 优化2. 并行处理路径标记，使用集合提升查询效率
        for key,path in path_pairs:
            try:
                # 生成路径边集合（使用集合提升查询效率）
                path_edges = set(zip(path[:-1],path[1:]))
                # 批量处理边标记
                for u,v in all_edges:
                    edge_exists = (u,v) in path_edges
                    # 使用NetworkX的安全属性设置方法
                    new_graph_data[u][v][key] = 1 if edge_exists else 0
            except Exception as e:
                print(f'[ERROR] 处理路径{key}时发生异常：{str(e)}')
                traceback.print_exc()

        return new_graph_data

        # try:
        #     for src in self.src_node_list:
        #         for dst in self.dst_node_list:
        #             key_name = f'({src}, {dst})'  # 键名
        #             # print('路径：',path_dict)
        #             _path = path_dict[key_name]  # 获取路径
        #             _path_set = [(_src, _dst) for _src, _dst in zip(_path[:-1], _path[1:])]
        #             print('路径集合:', _path_set)
        #             print('图边集合:', graph_data.edges())
        #             for u, v in graph_data.edges():
        #                 if (u, v) in _path_set:
        #                     new_graph_data[u][v][key_name] = 1
        #                 else:
        #                     new_graph_data[u][v][key_name] = 0
        # except Exception as e:
        #     print("[utils]路径赋值异常", e)
        # return new_graph_data

    async def networkx_to_dict(self, graph_data):
        """
        将图数据转化为字典
        :return:
        """
        return {
            "nodes": [{"id":n[0],**n[1]} for n in graph_data.nodes(data=True)],
            "edges": [{'source':u, "target":v, **d} for u,v,d in graph_data.edges(data=True)]
        }

    async def send_graph_data(self, graph_data):
        """
        发送数据的主函数
        :param graph_data:
        :return:
        """
        # 1.数据扩展
        new_graph_data = await self.send_graph_path_state(graph_data,self.path_file)
        # print('增广后的数据:', new_graph_data)
        new_graph_data_dict = await self.networkx_to_dict(new_graph_data)
        new_graph_data_dict['timestamp'] = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        dst_pro_path = setting.dst_pro_file_path
        if os.path.exists(dst_pro_path):
            with open(dst_pro_path, 'r', encoding='utf-8') as f:
                dst_pro_dict = json.load(f)
            new_graph_data_dict['action'] = dst_pro_dict['pro_dict']
            new_graph_data_dict['choice_key'] = dst_pro_dict['choice_key']
        else:
            new_graph_data_dict['action'] = None
            new_graph_data_dict['choice_key'] = None

        # 2.构建数据
        # response = requests.post(
        #     f'http://{self.server_ip}:{self.server_port}' + self.state_path,
        #     json = new_graph_data_dict,
        #     headers = {'Content-Type': 'application/json'}
        # )
        #
        # response = requests.post(
        #     f'http://{self.server_ip}:{self.server_port}' + self.request_path,
        #     json=new_graph_data_dict,
        #     headers={'Content-Type': 'application/json'}
        # )
        # 使用aiohttp实现正在的异步数据上传
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f'http://{self.server_ip}:{self.server_port}' + self.state_path,
                json = new_graph_data_dict,
                headers = {'Content-Type': 'application/json'}
            ) as resp1:
                pass
                # print(f'状态上传:{resp1.status}')

            async with session.post(
                    f'http://{self.server_ip}:{self.server_port}' + self.request_path,
                    json=new_graph_data_dict
            ) as resp2:
                pass
                # print(f'请求上传状态:{resp2.status}')


        self.send_graph_num += 1  # 发送图像计数
        print(f'第{self.send_graph_num}次数据上传成功！')


# 1. 定义发送类
class HubSendGraphData:
    def __init__(self,server_ip, server_port,state_path,request_path,action_path):
        self.server_ip = server_ip    # 服务端ip
        self.server_port = server_port   # 服务端端口
        self.state_path = state_path   # 服务端状态路径
        self.request_path = request_path   # 服务器请求路径
        self.action_path = action_path   # 服务端动作路径
        self.src_node_list = setting.src_node_list  # 源节点列表[1]
        self.dst_node_list = setting.dst_node_list  # 目的节点列表[15,16,17,18]
        self.path_file = setting.path_file_path  # 路径文件位置
        self.send_graph_num = 0

    def send_graph_path_state(self, graph_data, path_file):
        """
        在图属性的基础上添加路径的标识
        :param graph_data:
        :param path_file:
        :return:
        """
        if os.path.exists(path_file):
            try:
                with open(path_file, 'r', encoding='utf-8') as f:
                    path_dict = json.load(f)
                    # if f.read():
                    #     path_dict = json.load(f)
                    # else:
                    #     path_dict = None
            except json.JSONDecodeError as e:
                print(f'JSON文件读取错误{path_file}:{e}')
                path_dict = None

        else:
            path_dict = None

        new_graph_data = copy.deepcopy(graph_data)  # 对数据进行深拷贝，方便进行修改

        # 遍历边的信息
        """
        思路：将路径赋值成图的数据性
        1. 找到目标的路径：（1,15），（1,16），（1,17），（1,18）
        """
        # 优化1. 预生成所有需要标记的路径对
        path_pairs = []
        for src in self.src_node_list:
            for dst in self.dst_node_list:
                try:
                    # 统一键名格式（去除空格）
                    key = f'({src}, {dst})'
                    path = path_dict[key]
                    path_pairs.append((key,path))
                except Exception as e:
                    print(f'[WARN] 缺失路径配置：({src}, {dst}),错误：{e}')


        all_edges = list(graph_data.edges())
        # 优化2. 并行处理路径标记，使用集合提升查询效率
        for key,path in path_pairs:
            logger.debug(f"上传路径{path},时间{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}")
            path_edges = set(zip(path[:-1], path[1:]))
            for u,v in all_edges:
                new_graph_data[u][v][key] = 0
                if (u,v) in path_edges:
                    new_graph_data[u][v][key] = 1
                elif (v,u) in path_edges:
                    new_graph_data[u][v][key] = 1
                # else:
                #     print(f'边{(u,v)}在图中不存在，请检查')
                #     traceback.print_exc()
            logger.debug(f'增广后的数据{new_graph_data.edges(data=True)}')
            # try:
            #     # 生成路径边集合（使用集合提升查询效率）
            #     path_edges = set(zip(path[:-1],path[1:]))
            #     # 批量处理边标记
            #     for u,v in all_edges:
            #         edge_exists = (u,v) in path_edges
            #         # 使用NetworkX的安全属性设置方法
            #         new_graph_data[u][v][key] = 1 if edge_exists else 0
            # except Exception as e:
            #     print(f'[ERROR] 处理路径{key}时发生异常：{str(e)}')
            #     # traceback.print_exc()

        return new_graph_data

        # try:
        #     for src in self.src_node_list:
        #         for dst in self.dst_node_list:
        #             key_name = f'({src}, {dst})'  # 键名
        #             # print('路径：',path_dict)
        #             _path = path_dict[key_name]  # 获取路径
        #             _path_set = [(_src, _dst) for _src, _dst in zip(_path[:-1], _path[1:])]
        #             print('路径集合:', _path_set)
        #             print('图边集合:', graph_data.edges())
        #             for u, v in graph_data.edges():
        #                 if (u, v) in _path_set:
        #                     new_graph_data[u][v][key_name] = 1
        #                 else:
        #                     new_graph_data[u][v][key_name] = 0
        # except Exception as e:
        #     print("[utils]路径赋值异常", e)
        # return new_graph_data

    def networkx_to_dict(self, graph_data):
        """
        将图数据转化为字典
        :return:
        """
        return {
            "nodes": [{"id":n[0],**n[1]} for n in graph_data.nodes(data=True)],
            "edges": [{'source':u, "target":v, **d} for u,v,d in graph_data.edges(data=True)]
        }

    def send_graph_data(self, graph_data):
        """
        发送数据的主函数
        :param graph_data:
        :return:
        """
        # 1.数据扩展
        # new_graph_data = await self.send_graph_path_state(graph_data,self.path_file)
        new_graph_data = self.send_graph_path_state(graph_data, self.path_file)
        # print('增广后的数据:', new_graph_data)
        new_graph_data_dict = self.networkx_to_dict(new_graph_data)
        new_graph_data_dict['timestamp'] = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        dst_pro_path = setting.dst_pro_file_path
        if os.path.exists(dst_pro_path) and os.path.getsize(dst_pro_path):
            with open(dst_pro_path, 'r', encoding='utf-8') as f:
                dst_pro_dict = json.load(f)
            new_graph_data_dict['action'] = dst_pro_dict['pro_dict']
            new_graph_data_dict['choice_key'] = dst_pro_dict['choice_key']
        else:
            new_graph_data_dict['action'] = None
            new_graph_data_dict['choice_key'] = None

        # 2.构建数据
        response1 = requests.post(
            f'http://{self.server_ip}:{self.server_port}' + self.state_path,
            json = new_graph_data_dict,
            headers = {'Content-Type': 'application/json'}
        )

        response2 = requests.post(
            f'http://{self.server_ip}:{self.server_port}' + self.request_path,
            json=new_graph_data_dict,
            headers={'Content-Type': 'application/json'}
        )

        # # 使用aiohttp实现正在的异步数据上传
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(
        #         f'http://{self.server_ip}:{self.server_port}' + self.state_path,
        #         json = new_graph_data_dict,
        #         headers = {'Content-Type': 'application/json'}
        #     ) as resp1:
        #         print(f'状态上传:{resp1.status}')
        #
        #     async with session.post(
        #             f'http://{self.server_ip}:{self.server_port}' + self.request_path,
        #             json=new_graph_data_dict
        #     ) as resp2:
        #         print(f'请求上传状态:{resp2.status}')

        logger.debug(new_graph_data_dict)
        self.send_graph_num += 1  # 发送图像计数
        print(f'第{self.send_graph_num}次数据上传成功！')












if __name__ == '__main__':
    server_ip = '172.25.6.242'
    port = 5002
    client = SocketClient(server_ip=server_ip,port=port)
    # client.performance_test()
    G = client.create_test_graph()