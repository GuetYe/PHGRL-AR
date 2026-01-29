#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2025/4/7 14:35
@File:data_fraction.py
@Desc:数据拆分
"""
import logging
import re

import networkx as nx
import scipy.sparse as sp
from scipy.sparse import issparse
from scipy.sparse import coo_matrix, csc_matrix,csr_matrix,isspmatrix_csr,isspmatrix
import numpy as np
# 服务器将收集到的数据进行拆分放入经验池
import torch
import copy
import setting
from algorithms_model import gen_high_act_input_batch,gen_low_act_input_batch
from utils import log_management # calculate_node_order,
import random
import os

file_name = os.path.basename(__file__).split('.')[0]
logger = log_management(file_name, setting.print_level, setting.file_level)


class State:
    def __init__(self, sample_data):
        self.data = sample_data
        self.graph = self.get_graph()  # 图
        self.nodes = self.graph.nodes  # 节点
        self.edges = self.graph.edges  # 边
        self.adj = nx.adjacency_matrix(self.graph)  # 邻接矩阵
        self.norm_adj = self.sp_mx_norm(self.adj)  # 对称归一化后的邻接矩阵
        self.torch_norm_adj = self.sp_mx_to_torch_sp_tensor(self.norm_adj)
        self.trans_mx = self.create_trans_mx()
        self.torch_trans_mx = self.sp_mx_to_torch_sp_tensor(self.trans_mx)
        self.eadj, self.edge_name = self.create_edge_adj()  # 图边的邻接矩阵
        self.torch_norm_eadj = self.sp_mx_to_torch_sp_tensor(self.sp_mx_norm(self.eadj))  # 图边邻接矩阵tensor
        self.efeatures = self.get_efeatures()
        self.norm_efeatures = self.features_norm(self.efeatures)
        self.torch_norm_efeatures = self.sp_mx_to_torch_sp_tensor(self.norm_efeatures)
        self.edge_norm_feature_dict = self.efeatures_to_dict(self.norm_efeatures)
        self.features = self.get_features()
        self.norm_features = self.features_norm(self.features)
        self.torch_norm_features = self.sp_mx_to_torch_sp_tensor(self.norm_features)
        self.nodes_order_tensor = None  # 初始为空列表，在下层状态中才会出现
        self.node_order_dict = self.generate_node_order(setting.ordering_attr_list)

    def generate_node_order(self, attr_list):
        """
        根据目的生成节点的序
        :param attr_list:
        :return:
        """
        if isinstance(attr_list,list):
            weight = attr_list[0]   # 这里可以添加其他逻辑处理多属性的形式，这里简化处理
        else:
            weight = attr_list

        # 初始化没对节点之间的最短权重
        shortest_path_matrix = np.zeros((len(self.nodes),len(self.nodes)))

        # 遍历没对节点
        for index1,u in enumerate(self.nodes):
            for index2,v in enumerate(self.nodes):
                if u != v:
                    # 获取最短路径（使用Dijkstra算法）
                    path = nx.dijkstra_path(self.graph, source=u, target=v, weight=weight)
                    path_weight_sum = sum(self.graph[u][v][weight] for u,v in zip(path[:-1], path[1:]))
                    shortest_path_matrix[index1, index2] = path_weight_sum
                else:
                    shortest_path_matrix[index1, index2] = 0  # 自迭代距离为0

        # 打印节点顺序
        print("节点顺序",list(self.nodes))
        # 提取目的列的数据
        node_order_dict = {}  # 目的节点序列列表
        for dst in setting.dst_list:  # 遍历目的节点列表
            shortest_path_value = shortest_path_matrix[:, list(self.nodes).index(dst)].ravel()

            # 绑定节点序列和路径值序列
            zipped = list(zip(list(self.nodes), shortest_path_value))

            # 排序
            zipped_sort = sorted(zipped, key=lambda x:x[1], reverse=True)

            # 排序结果
            node_sort, value_sort = zip(*zipped_sort)

            node_order_dict[dst] = node_sort

        return node_order_dict


    def ref_sort(self,to_sort_list,reference):
        """
        根据给定的参考序列排序
        :param to_sort_list:
        :param reference:
        :return:
        """
        # 建立元素到索引的映射
        ordes_dict = {item:idx for idx, item in enumerate(reference)}

        # 指定默认权重，放在末尾
        default_pos = len(reference)

        # 排序
        sorted_list = sorted(to_sort_list, key=lambda x: ordes_dict.get(x, default_pos))

        return sorted_list




    def get_graph(self):
        """
        获取节点信息
        :return:
        """
        graph = nx.Graph()
        for n in self.data['nodes']:
            graph.add_node(n['id'], **{k: v for k, v in n.items() if k != 'id'})  # 反序列节点属性
        for e in self.data['edges']:
            graph.add_edge(e['source'], e['target'],
                           **{k: v for k, v in e.items() if k not in ['source', 'target']})  # 反序列边属性
        return graph

    def sp_mx_norm(self, mx):
        """
        稀疏矩阵的对称归一化
        :return:
        """
        # 安全检查
        if not isspmatrix_csr(mx):
            mx = mx.tocsr() if isspmatrix(mx) else csr_matrix(mx)

        # 1.添加自环避免除零错误
        mx_self = mx + sp.eye(mx.shape[0])

        # 2.计算度矩阵D
        try:
            degrees = mx_self.sum(axis=1).A1
        except AttributeError:
            degrees = np.asarray(mx_self.sum(axis=1)).flatten()

        # 数值有效性验证
        if np.any(degrees<=0):
            invalid_nodes = np.where(degrees<=0)[0]
            raise ValueError(f'节点{invalid_nodes}的度非正，请检查输入矩阵')


        # 3.计算D^(-1/2)
        D_inv_sqrt = sp.diags(1.0 / (degrees ** 0.5),offsets=0,format='csr')

        # 4.对称归一化
        mx_norm = D_inv_sqrt.dot(mx_self).dot(D_inv_sqrt)

        return mx_norm

    def sp_mx_to_torch_sp_tensor(self, sp_mx):
        """
        将scipy稀疏矩阵转化为torch的稀疏矩阵形式
        :param sp_mx:
        :return:
        """
        if sp_mx is None:
            return None

        sp_mx = sp_mx.tocoo().astype(np.float32)  # 将格式转化为COO格式
        indices = torch.from_numpy(np.vstack((sp_mx.row, sp_mx.col)).astype(np.int64))  # 构建索引张量
        values = torch.from_numpy(sp_mx.data)  # 构建数据值张量
        shape = torch.Size(sp_mx.shape)  # 构建稀疏张量
        return torch.sparse_coo_tensor(indices, values, shape).coalesce().to(torch.float32)

    def create_trans_mx(self):
        """
        创建转移矩阵N_v * N_e
        :return:
        """
        vertex_adj = copy.deepcopy(self.adj)
        # 1. 消除自环边
        vertex_adj.setdiag(0)
        # 2. 提取无向边列表，只保留上三角部分避免重复
        edge_index = np.nonzero(sp.triu(vertex_adj, k=1))

        # 3. 计算无向边数量
        num_edge = int(len(edge_index[0]))

        # 4. 生成边列表
        # 例如：[(0,1), (0,2), (1,2)] 表示三条边
        edge_name = [x for x in zip(edge_index[0], edge_index[1])]

        # 5. 构建转移矩阵的行索引
        # 例如边列表[(0,1), (0,2)] 转换为 [0,1,0,2]
        row_index = [i for sub in edge_name for i in sub]

        # 6. 构建转移矩阵列索引
        col_index = np.repeat([i for i in range(num_edge)], 2)

        # 7.生成填充数据（所有元素为1）
        data = np.ones(num_edge * 2)

        # 8.构造CSR格式的稀疏矩阵
        T = sp.csr_matrix((data, (row_index, col_index)), shape=(vertex_adj.shape[0], num_edge))

        return T

    def create_edge_adj(self):
        """
        通过节点邻接矩阵创建边的的邻接矩阵
        :return:
        """
        vertex_adj = copy.deepcopy(self.adj)
        vertex_adj.setdiag(0)  # 去回环
        edge_index = np.nonzero(sp.triu(vertex_adj, k=1))
        num_edge = int(len(edge_index[0]))
        edge_name = [x for x in zip(edge_index[0], edge_index[1])]

        edge_adj = np.zeros((num_edge, num_edge))  # 初始化
        for i in range(num_edge):
            for j in range(i, num_edge):
                if len(set(edge_name[i]) & set(edge_name[j])) == 0:  # 边没有交点，不相邻
                    edge_adj[i, j] = 0
                else:
                    edge_adj[i, j] = 1

        adj = edge_adj + edge_adj.T  # 将上三角函数变成对称矩阵
        np.fill_diagonal(adj, 1)  # 用1填充对角线
        return sp.csr_matrix(adj), edge_name  # 返回csr矩阵和边名称

    def get_efeatures(self):
        """
        获取边的特征csr矩阵
        :return:
        """
        # 1.获取所有边的属性名称
        # all_attrs = set()
        # for _,_,d in self.graph.edges(data=True):
        #     all_attrs.update(d.keys())
        #
        # columns = sorted(all_attrs)  # 固定属性内容及相应的顺序
        # ['(1, 15)', '(1, 16)', '(1, 17)', '(1, 18)', 'delay', 'distance', 'forward_queue_bytes', 'forward_queue_pkts',
        # 'free_bw', 'loss', 'pkt_drop', 'pkt_err', 'reverse_queue_bytes', 'reverse_queue_pkts', 'used_bw']
        # print(columns)
        choice_attrs = setting.edge_attr_choice
        columns = choice_attrs

        # # 2. 构建行索引映射
        # edge_index = {(u,v):idx for idx, (u,v) in enumerate(self.graph.edges())}

        # 3. 创建COO格式稀疏矩阵
        rows, cols, data = [], [], []
        for idx, (u, v, attrs) in enumerate(self.graph.edges(data=True)):
            for attr in columns:
                if attr in attrs:
                    rows.append(idx)
                    cols.append(columns.index(attr))
                    data.append(attrs[attr])

        mx = sp.coo_matrix((data, (rows, cols)),
                           shape=(len(self.graph.edges()), len(columns)))
        return mx

    def features_norm(self, coo_data):
        """
        特征矩阵归一化，要求特征矩阵的格式是coo类型的
        :return:
        """
        # # 测试
        # row = np.array([0, 0, 1, 2])
        # col = np.array([0, 2, 1, 2])
        # data = np.array([1, 4, 3, 5])
        # coo_data = coo_matrix((data, (row, col)), shape=(3, 3))
        if coo_data is None:
            return None

        # 1.将COO格式的数据转化为CSC格式
        csc = coo_data.tocsc()

        # 2.按列计算极值
        col_min = np.zeros(csc.shape[1])
        col_max = np.zeros(csc.shape[1])

        for j in range(csc.shape[1]):
            col_slice = csc[:, j]
            # start = csc.indptr[j]
            # end = csc.indptr[j+1]
            # if start != end:
            if col_slice.nnz > 0:
                # col_min[j] = csc[start:end].min()
                # col_max[j] = csc[start:end].max()
                col_min[j] = col_slice.min()
                col_max[j] = col_slice.max()

        # 3.避免除零错误
        ranges = col_max - col_min
        valid_ranges = ranges > 0
        scale = np.divide(1, ranges, where=valid_ranges,
                          out=np.zeros_like(ranges))

        # 4.执行向量归一化
        # norm_data = (csc.data-col_min[csc.indices])*scale[csc.indices]
        norm_data = np.zeros_like(csc.data)
        for j in range(csc.shape[1]):
            start = csc.indptr[j]
            end = csc.indptr[j + 1]
            if valid_ranges[j]:
                norm_data[start:end] = (csc.data[start:end] - col_min[j]) * scale[j]

        # 5.重建COO矩阵
        norm_data_coo = csc.tocoo()
        norm_data_coo.data = norm_data

        return norm_data_coo

    def efeatures_to_dict(self, state_mx):
        """
        将边的特征矩阵转化为字典类型
        :param state_mx:
        :return:
        """
        np_state_mx = state_mx.toarray()
        edge_feature_dict = {}
        for i in range(len(self.edge_name)):
            edge_feature_dict[self.edge_name[i]] = torch.from_numpy(np_state_mx[i, :])
        return edge_feature_dict

    def get_features(self):
        """
        提取节点上的属性特征
        :return:
        """
        choice_attrs = setting.node_attr_choice
        columns = choice_attrs
        if not all(dict(self.graph.nodes(data=True)).values()):
            return None

        # # 2. 构建行索引映射
        # edge_index = {(u,v):idx for idx, (u,v) in enumerate(self.graph.edges())}

        # print(self.graph.nodes(data=True))

        # 3. 创建COO格式稀疏矩阵
        rows, cols, data = [], [], []
        for idx, (node, attr_dict) in enumerate(self.graph.nodes(data=True)):
            for attr in columns:
                if attr in attr_dict.keys():
                    rows.append(idx)
                    cols.append(columns.index(attr))
                    data.append(attr_dict[attr])

        mx = sp.coo_matrix((data, (rows, cols)),
                           shape=(len(self.graph.nodes()), len(columns)))
        # print('mx', mx)
        return mx

    def get_low_state_action_reward(self, cur_site, dst_site, next_site):
        """
        获取下层状态
        :param cur_site: 当前位置
        :param dst_site: 目标位置
        :param next_site: 下一个位置
        :return:
        """
        # 1. 首先更新节点属性编码
        new_state = copy.deepcopy(self)
        low_action = {}  # 用字典的形式保存动作
        # 标识做排序，避免交集为空的问题
        _cur_site, _next_site = self.ref_sort([cur_site, next_site], self.node_order_dict[dst_site])
        # print(f'排序前源{cur_site}，目的{next_site}')
        # print(f'排序后源{_cur_site}, 目的{_next_site}')

        for node in new_state.graph.nodes():
            if node == _cur_site:
                new_state.graph.nodes[node]['cur_site'] = 1
            else:
                new_state.graph.nodes[node]['cur_site'] = 0

            if node == dst_site:
                new_state.graph.nodes[node]['dst_site'] = 1
            else:
                new_state.graph.nodes[node]['dst_site'] = 0

            if node == _next_site:
                low_action[node] = 1
            else:
                low_action[node] = 0

        # 3. 重置节点特征
        new_state.features = new_state.get_features()
        new_state.norm_features = new_state.features_norm(new_state.features)
        new_state.torch_norm_features = new_state.sp_mx_to_torch_sp_tensor(new_state.norm_features)
        # print(f'节点特征矩阵:{new_state.torch_norm_features.to_dense().float().unsqueeze(0)}')

        # print(new_state.graph.nodes(data=True))

        # 4. 计算下层奖励
        step_reward = new_state.get_low_reward(cur_site, next_site)
        if _next_site == dst_site:  # 这里同样要算到达目的地奖励
            step_reward += setting.terminal_reward

        # 计算目标序
        # dst_order, _ = calculate_node_order(new_state.graph, cur_site, dst_site, weight_list=setting.ordering_attr_list)
        dst_order = self.node_order_dict[dst_site]
        # print(f'修改后的节点排序{dst_order}')

        raw_nodes = dst_order[dst_order.index(_cur_site):]  # 这里的排序也得是修正后的
        new_state.nodes_order_tensor = torch.from_numpy(
            np.array([1 if node in raw_nodes else 0 for node in new_state.graph.nodes]))
        # print(f"节点序：{new_state.nodes_order_tensor}")

        # print(f"下层单步奖励{step_reward}")

        # 5. 计算下一个状态
        next_state = new_state.get_next_state(_next_site)

        return new_state, low_action, step_reward, next_state


    # 获取当前状态
    def get_low_cur_state(self, cur_site, dst_site):
        """
        获取下层状态 -- 修改状态的值
        :param cur_site: 当前位置
        :param dst_site: 目标位置
        :return:
        """
        # 1. 首先更新节点属性编码
        new_state = copy.deepcopy(self)
        for node in new_state.graph.nodes():
            if node == cur_site:
                new_state.graph.nodes[node]['cur_site'] = 1
            else:
                new_state.graph.nodes[node]['cur_site'] = 0

            if node == dst_site:
                new_state.graph.nodes[node]['dst_site'] = 1
            else:
                new_state.graph.nodes[node]['dst_site'] = 0

        # 3. 重置节点特征
        new_state.features = new_state.get_features()
        new_state.norm_features = new_state.features_norm(new_state.features)
        new_state.torch_norm_features = new_state.sp_mx_to_torch_sp_tensor(new_state.norm_features)
        #  print(f'节点特征矩阵:{new_state.torch_norm_features}')


        # 计算目标序
        # dst_order, _ = calculate_node_order(new_state.graph,cur_site,dst_site, weight_list=setting.ordering_attr_list)
        dst_order = self.node_order_dict[dst_site]
        # print(f'修改后的节点排序{dst_order}')

        # if dst_site == 17:
        #     print(f"目标节点为{dst_site}的序大小{dst_order}")
        raw_nodes = dst_order[dst_order.index(cur_site):]
        # print(f"节点序{raw_nodes}")
        new_state.nodes_order_tensor = torch.from_numpy(np.array([1 if node in raw_nodes else 0 for node in new_state.graph.nodes]))
        # print(f"节点序：{new_state.nodes_order_tensor}")
        return new_state

    def get_low_sample(self):
        """
        获取下层样本
        :return: 样本序列
        """
        sample_list = []   # 样本序列
        src_dst_pair = setting.src_dst_pair
        for src_dst in src_dst_pair:
            dst = int(re.findall(r'\d+',src_dst)[-1])
            for u,v,attr_dict in self.graph.edges(data=True):
                if attr_dict[src_dst] == 1:
                    state, low_action, step_reward, next_state = self.get_low_state_action_reward(u,dst,v)
                    if v == dst:
                        done = True
                    else:
                        done = False
                    sample_list.append((state,low_action, step_reward, next_state,done))
        return sample_list



    def get_next_state(self, next_site):
        """
        获取下一个状态
        :param next_site:
        :return:
        """
        next_state = copy.deepcopy(self)
        for node in next_state.nodes():
            if node == next_site:
                next_state.graph.nodes[node]['cur_site'] = 1
            else:
                next_state.graph.nodes[node]['cur_site'] = 0
        return next_state

    def get_low_reward(self, cur_site, next_site):
        """
        计算单步奖励(负值)
        :param cur_site: 当前位置
        :param next_site: 下一跳位置
        :return:
        """
        choice_attrs = setting.edge_attr_choice
        reward_list = []  # 记录奖励
        attr_min_list = []  # 记录最小值，方便进行归一化
        attr_max_list = []  # 记录最大值，方便进行归一化
        tag = 0
        for u, v, attr_dict in self.graph.edges(data=True):
            if tag == 0:
                for index, attr_name in enumerate(choice_attrs):
                    attr_min_list.append(attr_dict[attr_name])
                    attr_max_list.append(attr_dict[attr_name])
                tag = 1
            else:
                for index, attr_name in enumerate(choice_attrs):
                    attr_min_list[index] = min(attr_dict[attr_name], attr_min_list[index])
                    attr_max_list[index] = max(attr_dict[attr_name], attr_max_list[index])

            if u == cur_site and v == next_site:
                for attr_name in choice_attrs:
                    reward_list.append(attr_dict[attr_name])  # 记录边的奖励，可能不存在，得进行特殊处

        # 归一化
        if reward_list == []:
            # print(f'边{cur_site}-{next_site}没找到，进行反向查找！')
            choice_attrs = setting.rev_edge_attr_choice
            for u, v, attr_dict in self.graph.edges(data=True):
                if tag == 0:
                    for index, attr_name in enumerate(choice_attrs):
                        attr_min_list.append(attr_dict[attr_name])
                        attr_max_list.append(attr_dict[attr_name])
                    tag = 1
                else:
                    for index, attr_name in enumerate(choice_attrs):
                        attr_min_list[index] = min(attr_dict[attr_name], attr_min_list[index])
                        attr_max_list[index] = max(attr_dict[attr_name], attr_max_list[index])

                if u == next_site and v == cur_site:
                    for attr_name in choice_attrs:
                        reward_list.append(attr_dict[attr_name])  # 记录边的奖励，可能不存在，得进行特殊处

        if reward_list == []:
            # print(f'边{cur_site}-{next_site}没找到，正反向均为发现！')
            print(self.graph.edges(data=True))
            return None

        norm_reward_list = (np.array(reward_list) - np.array(attr_min_list)) / (
                    np.array(attr_max_list) - np.array(attr_min_list) + 1e-8)

        # print('下层归一化奖励', norm_reward_list)

        reward_type = setting.reward_type
        for index, attr_type in enumerate(reward_type):
            if attr_type == 'max':
                norm_reward_list[index] = norm_reward_list[index]-1  # 避免很小，接近于零
            elif attr_type == 'min':
                norm_reward_list[index] = - norm_reward_list[index]

        reward_value = np.dot(norm_reward_list, setting.bate)
        # if reward_value < -200:
        #     print('norm_reward_list',norm_reward_list)

        # logger.debug(f'下层各属性的奖励：{norm_reward_list}')

        return reward_value

    # def nodes_ordering(self):
    #     """
    #     通过序变量确定节点序
    #     :return:
    #     """
    #     src_list = setting.src_list
    #     dst_list = setting.dst_list
    #     weight_list = [setting.edge_attr_choice[index] for index in setting.ordering_attr_list]
    #     nodes_order = {}
    #     for src in src_list:
    #         for dst,key in zip(dst_list,setting.src_dst_pair):
    #             nodes_order[key] = calculate_node_order(self.graph,src,dst,weight_list)
    #     # print(f"节点序:{nodes_order}")
    #     return nodes_order


def get_high_action_pro(sample_data):
    """
    获取上层动作
    :param sample_data:
    :return:
    """
    if setting.high_train_model == 'PPO':
        choice_key = sample_data['choice_key']
        if choice_key == None:
            print('[data_fraction.py]还没有动作数据，用随机数代替')
            action = random.sample(range(len(setting.src_dst_pair)),k=1)[0]
        else:
            action = setting.src_dst_pair.index(choice_key)

        return action
    else:
        action_dict = sample_data['action']
        action_seq = setting.src_dst_pair
        action_prob = []  # 用于存储任播动作概率
        if action_dict == None:
            print('[data_fraction.py]还没有动作数据，用平均数代替')
            for src_dst in action_seq:
                action_prob.append(1./len(action_seq))
        else:
            for src_dst in action_seq:
                action_prob.append(action_dict[src_dst])

        return action_prob


def get_high_reward_list(sample_data):
    """
    获取上层奖励，根据不同的源和目的对计算奖励
    :param sample_data:
    :return:
    """
    action_seq = setting.src_dst_pair
    attrs = setting.edge_attr_choice
    attr_types = setting.edge_attr_type
    reward_types = setting.reward_type
    # print(f'[data_fraction.py]运算前，样本数据{sample_data}。')  # 数据已经出现了问题

    reword_dict = {}
    for src_dst in action_seq:
        # 根据不同特征类型的计算不同源-目的对的奖励，还需要注意归一化，记录最大最小值
        src_dst_reward_list = []
        for attr, attr_type, reward_type in zip(attrs, attr_types, reward_types):
            # print('属性',attr)
            sub_reword = compute_attr_norm_reward(sample_data, src_dst, attr, attr_type, reward_type)
            src_dst_reward_list.append(sub_reword)
        reword_dict[src_dst] = src_dst_reward_list
    return reword_dict


def compute_attr_norm_reward(sample_data, src_dst, attr, attr_type, reward_type):
    """
    通过样本数据计算指定目标归一化后的奖励值
    :param sample_data: 样本数据
    :param src_dst: 源-目标
    :param attr: 属性
    :param attr_type: 属性类型
    :param reward_type: 奖励类型
    :return:
    """
    # print(f'[data_fraction.py]运算前，样本数据{sample_data}。')  # 这里数据有问题
    attr_reward = None  # 初始化奖励，用于记录奖励值，这里不方便给定是0和1，因为不同属性类型的计算不一样，所以初始化为None
    attr_min = float('inf')  # 初始化属性最小值，用于记录属性最小值
    attr_max = float('-inf')  # 初始化属性最大值，用于记录属性最大值
    compute_tag = 0  # 计算标记，用于控制计算操作

    # 从样本数据中提取奖励值
    for e in sample_data['edges']:
        # 判断该边是否在源和目的的路径上
        if e[src_dst] == 0:  # 不在的情况，仅仅更新最大值和最小值
            attr_min = min(attr_min, e[attr])
            attr_max = max(attr_max, e[attr])

        else:  # 在的情况，除了更新最大值和最小值外，还得根据属性类型计算奖励
            attr_min = min(attr_min, e[attr])
            attr_max = max(attr_max, e[attr])
            # print(src_dst)

            if compute_tag == 0:  # 首次计算
                attr_reward = e[attr]*e[src_dst]  # 考虑到环路或者重复路径的形式
                # print(f'初始值：{attr_reward},属性{attr}, 边{e}，属性值：{e[attr]}，是否计算属性{e[src_dst]}')
                compute_tag = 1
            else:
                if attr_type == 'MIN':
                    attr_reward = min(attr_reward, e[attr]*e[src_dst])
                elif attr_type == 'PROD':
                    # 例子，拿着的永远是丢包率，迭代更新
                    attr_reward = 1 - (1 - attr_reward) * (1 - e[attr]*e[src_dst])
                    # print(f'计算结果{attr_reward}')
                elif attr_type == 'FSUM':
                    attr_reward += e[attr]*e[src_dst]

    # 计算结束后，进行归一化, 注意排除除零的情况
    # print('属性值',attr_reward)
    # print('最大值',attr_max)
    # print('最小值',attr_min)
    if attr_max == attr_min:
        attr_reward = 0
    elif attr_reward == None:
        print(f'[data_fraction.py]路径度量计算出现问题{attr_reward},反馈极端值。')
        print(f'[data_fraction.py]检查样本数据{sample_data}。')
        if reward_type == 'max':
            attr_reward = 0
        elif reward_type == 'min':
            attr_reward = - 1
        return attr_reward

    else:
        if reward_type == 'max':
            attr_reward = (attr_reward - attr_max) / (attr_max - attr_min)
        elif reward_type == 'min':
            attr_reward = -(attr_reward - attr_min) / (attr_max - attr_min)

    # if reward_type == 'max':
    #     # attr_reward = -1 / attr_reward if attr_reward > 0 else 0
    #     attr_reward = (attr_reward-attr_max) / (attr_max - attr_min)
    # elif reward_type == 'min':
    #     attr_reward = - attr_reward
    # print(f'属性{attr}奖励{attr_reward}')
    return attr_reward


# 计算奖励
def get_high_reward(sample_data, action_pro):
    """
    计算奖励
    :param sample_data: 状态
    :param action_pro: 动作
    :return:
    """
    # logger.debug(f'运算奖励之前的状态：{sample_data}') # 这里也正常
    # print(f'[data_fraction.py]运算前，样本数据{sample_data}。')
    sub_reward_dict = get_high_reward_list(sample_data)
    # logger.debug(f'sub_reward_dict:{sub_reward_dict}')
    reward_list = []  # 初始化奖励函数
    for src_dst in setting.src_dst_pair:
        src_dst_reward = np.dot(sub_reward_dict[src_dst], setting.bate)
        reward_list.append(src_dst_reward)


    if setting.high_train_model =='PPO':
        # print(f'动作{action_pro}')
        return reward_list[action_pro]
    else:
        # print(f'奖励列表{reward_list}')
        # print(f'相应概率{action_pro}')
        if np.dot(reward_list, action_pro) > 0:
            logger.info(f'奖励列表:{reward_list}\n相应概率:{action_pro}\nsub_reward_dict:{sub_reward_dict}')
        return np.dot(reward_list, action_pro)

def take_low_agent_act(low_agent,state,evaluate=False):
    """
    获取下层智能体的动作
    :param low_agent: 下层智能体
    :param state: 状态
    :return: new_state_data,path_dict,reward_dict 新状态数据，路径集合，奖励结合
    """
    # 数据拷贝
    new_state = copy.deepcopy(state)
    new_state_data = copy.deepcopy(state.data)

    # 遍历源节点和目的节点
    path_dict = {}
    reward_dict = {}
    for src in setting.src_list:
        for dst, src_dst in zip(setting.dst_list,setting.src_dst_pair):    # 这样写的局限是只有一个源节点，多个目的节点
            done = False
            cur_site = copy.deepcopy(src)
            dst_site = copy.deepcopy(dst)
            path = [cur_site]   # 用于保存路径
            reward = 0   # 用于保存奖励
            num_steps = 0

            while not done or num_steps>setting.max_evaluate_step:
                _cul_low_state = new_state.get_low_cur_state(cur_site=cur_site, dst_site=dst_site)
                # print(f'当前节点{cur_site},目的节点{dst_site}节点顺序{list(_cul_low_state.nodes)}节点的序{_cul_low_state.nodes_order_tensor}')
                _low_act_input = gen_low_act_input_batch([_cul_low_state]).to(setting.device)
                # print(f'节点的序{_low_act_input.Mask}')

                _low_act_input_copy = copy.deepcopy(_low_act_input)
                # print(f"_low_act_input_copy id{id(_low_act_input_copy)}")
                _act = low_agent.actor(_low_act_input_copy, evaluate=evaluate)
                next_site = list(new_state.nodes)[_act.tolist()[0][0].index(1)]  # 获取下一个节点的位置
                path.append(next_site)   # 更新路径
                _, _, step_reward, _ = new_state.get_low_state_action_reward(cur_site,dst_site,next_site)
                reward += step_reward  # 更新奖励
                cur_site = next_site  # 更新当前位置
                if cur_site == dst_site:
                    done = True
            else:
                if not done:
                    reward += setting.terminal_punishment  # 添加未到达终点的惩罚
            # if dst == 17:
            #     print(f'目的节点{dst}的路径为{path}')

            path_dict[src_dst] = path   # 更新路径字典
            reward_dict[src_dst] = reward  # 更新奖励字典
            for index,edge in enumerate(new_state_data['edges']):
                new_state_data['edges'][index][src_dst] = 0   # 全部清零

            for u, v in zip(path[:-1], path[1:]):
                for index,edge in enumerate(new_state_data['edges']):
                    if (edge['source'] == u and edge['target'] == v) or (edge['source'] == v and edge['target'] == u):
                        new_state_data['edges'][index][src_dst] += 1   # 注意到存在环路的情况
                    # else:
                    #     edge[src_dst] = 0
        # print(f'修改后的数据{new_state_data}')

    return new_state_data, path_dict, reward_dict


if __name__ == '__main__':
    # sample_data = {'edges': [
    #     {'(1, 15)': 1, '(1, 16)': 1, '(1, 17)': 1, '(1, 18)': 1, 'delay': 650.6130695343018, 'distance': 50.0,
    #      'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 38.99975903614458, 'loss': 0.5401998739533628,
    #      'pkt_drop': 1, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 1, 'target': 5,
    #      'used_bw': 0.0002409638554216868},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 0, '(1, 18)': 0, 'delay': 902.21107006073, 'distance': 70.71,
    #      'forward_queue_bytes': 60, 'forward_queue_pkts': 1, 'free_bw': 37.99949774586742, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 5, 'target': 2,
    #      'used_bw': 0.0005022541325763897},
    #     {'(1, 15)': 1, '(1, 16)': 1, '(1, 17)': 1, '(1, 18)': 1, 'delay': 701.261043548584, 'distance': 200.0,
    #      'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 39.999683638161144, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 5, 'target': 11,
    #      'used_bw': 0.0003163618388531889},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 0, '(1, 18)': 0, 'delay': 941.26296043396, 'distance': 50.0,
    #      'forward_queue_bytes': 98, 'forward_queue_pkts': 1, 'free_bw': 4.999458796219533, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 2, 'target': 6,
    #      'used_bw': 0.0005412037804675837},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 0, '(1, 18)': 0, 'delay': 965.1069641113281, 'distance': 70.71,
    #      'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 39.99953838501078, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 3, 'target': 6,
    #      'used_bw': 0.00046161498922235074},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 0, '(1, 18)': 0, 'delay': 974.4843244552612, 'distance': 50.0,
    #      'forward_queue_bytes': 60, 'forward_queue_pkts': 1, 'free_bw': 6.999632396694214, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 60, 'reverse_queue_pkts': 1, 'source': 3, 'target': 7,
    #      'used_bw': 0.000367603305785125},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 0, '(1, 18)': 0, 'delay': 825.6047964096069, 'distance': 50.0,
    #      'forward_queue_bytes': 60, 'forward_queue_pkts': 1, 'free_bw': 38.99984126984127, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 70, 'reverse_queue_pkts': 1, 'source': 6, 'target': 9,
    #      'used_bw': 0.0001587301587301594},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 0, '(1, 18)': 0, 'delay': 636.7499828338623, 'distance': 70.71,
    #      'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 30.999410247933884, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 7, 'target': 4,
    #      'used_bw': 0.0005897520661157041},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 0, '(1, 18)': 0, 'delay': 722.9998111724854, 'distance': 70.71,
    #      'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 4.9995900826446285, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 7, 'target': 9,
    #      'used_bw': 0.0004099173553719},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 1, '(1, 18)': 1, 'delay': 968.2302474975586, 'distance': 100.0,
    #      'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 30.9996830637174, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 7, 'target': 10,
    #      'used_bw': 0.00031693628260151904},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 0, '(1, 18)': 0, 'delay': 734.8668575286865, 'distance': 50.0,
    #      'forward_queue_bytes': 60, 'forward_queue_pkts': 1, 'free_bw': 34.99976206212822, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 4, 'target': 8,
    #      'used_bw': 0.00023793787177792496},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 0, '(1, 18)': 0, 'delay': 795.5363988876343, 'distance': 111.8,
    #      'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 10.999683063717399, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 8, 'target': 10,
    #      'used_bw': 0.00031693628260151904},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 0, '(1, 18)': 0, 'delay': 892.2604322433472, 'distance': 300.0,
    #      'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 29.999683950617285, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 8, 'target': 14,
    #      'used_bw': 0.0003160493827160485},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 0, '(1, 18)': 0, 'delay': 823.532223701477, 'distance': 158.11,
    #      'forward_queue_bytes': 60, 'forward_queue_pkts': 1, 'free_bw': 21.999762689518786, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 9, 'target': 11,
    #      'used_bw': 0.00023731048121291976},
    #     {'(1, 15)': 0, '(1, 16)': 1, '(1, 17)': 0, '(1, 18)': 0, 'delay': 1004.6427249908447, 'distance': 100.0,
    #      'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 12.999683481701286, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 9, 'target': 12,
    #      'used_bw': 0.0003165182987141443},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 0, '(1, 18)': 0, 'delay': 781.207799911499, 'distance': 70.71,
    #      'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 14.999683481701286, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 10, 'target': 12,
    #      'used_bw': 0.0003165182987141443},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 1, '(1, 18)': 0, 'delay': 916.9350862503052, 'distance': 200.0,
    #      'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 12.999683690280065, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 10, 'target': 13,
    #      'used_bw': 0.000316309719934101},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 0, '(1, 18)': 1, 'delay': 646.3301181793213, 'distance': 206.16,
    #      'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 25.999683950617285, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 10, 'target': 14,
    #      'used_bw': 0.00031604938271605},
    #     {'(1, 15)': 1, '(1, 16)': 0, '(1, 17)': 0, '(1, 18)': 0, 'delay': 757.4924230575562, 'distance': 50.0,
    #      'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 33.99976276771005, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 11, 'target': 15,
    #      'used_bw': 0.00023723228995057688},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 0, '(1, 18)': 0, 'delay': 779.4176340103149, 'distance': 111.8,
    #      'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 17.999591433278418, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 12, 'target': 15,
    #      'used_bw': 0.0004085667215815491},
    #     {'(1, 15)': 0, '(1, 16)': 1, '(1, 17)': 0, '(1, 18)': 0, 'delay': 849.0314483642578, 'distance': 150.0,
    #      'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 13.999591635106208, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 12, 'target': 16,
    #      'used_bw': 0.0004083648937921944},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 0, '(1, 18)': 0, 'delay': 862.714409828186, 'distance': 50.0,
    #      'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 20.999683846533838, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 13, 'target': 16,
    #      'used_bw': 0.00031615346616169893},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 1, '(1, 18)': 0, 'delay': 899.1731405258179, 'distance': 150.0,
    #      'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 8.999684158578713, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 13, 'target': 17,
    #      'used_bw': 0.00031584142128639534},
    #     {'(1, 15)': 0, '(1, 16)': 0, '(1, 17)': 0, '(1, 18)': 1, 'delay': 601.8701791763306, 'distance': 50.0,
    #      'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 22.999670781893006, 'loss': 0.0, 'pkt_drop': 0,
    #      'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 14, 'target': 18,
    #      'used_bw': 0.0003292181069958838}],
    #                'nodes': [{'id': 1}, {'id': 5}, {'id': 2}, {'id': 3}, {'id': 6}, {'id': 7}, {'id': 4}, {'id': 8},
    #                          {'id': 9}, {'id': 10}, {'id': 11}, {'id': 12}, {'id': 13}, {'id': 14}, {'id': 15},
    #                          {'id': 16}, {'id': 17}, {'id': 18}], 'timestamp': '2025-04-06-11-53-18',
    #                'action': {'(1, 15)': 0.3, '(1, 16)': 0.22, '(1, 17)': 0.23, '(1, 18)': 0.25}}
    # sample_data = {'action': {'(3, 15)': 2.1684445528080687e-07, '(3, 16)': 0.9859153032302856, '(3, 17)': 0.014084492810070515, '(3, 18)': 5.010304948704913e-10}, 'choice_key': '(3, 16)', 'edges': [{'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 3.6191940307617188, 'distance': 10.0, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 471.9996864282215, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 5, 'target': 1, 'used_bw': 0.000313571778539929}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 2.9451847076416016, 'distance': 20.0, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 486.99976900866216, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 5, 'target': 11, 'used_bw': 0.0002309913378248344}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 3.6362409591674805, 'distance': 14.14, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 464.9997648211661, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 5, 'target': 2, 'used_bw': 0.00023517883390496418}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 3.2067298889160156, 'distance': 10.0, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 419.99968647942524, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 6, 'target': 2, 'used_bw': 0.0003135205747877076}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 2.8312206268310547, 'distance': 10.0, 'forward_queue_bytes': 3028, 'forward_queue_pkts': 2, 'free_bw': 465.5304443360959, 'loss': 0.01999930907356237, 'pkt_drop': 162, 'pkt_err': 0.0, 'reverse_queue_bytes': 66, 'reverse_queue_pkts': 1, 'source': 6, 'target': 9, 'used_bw': 9.469555663904107}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 3.1702518463134766, 'distance': 14.14, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 209.52342932195026, 'loss': 0.01666878928204303, 'pkt_drop': 135, 'pkt_err': 0.0, 'reverse_queue_bytes': 4542, 'reverse_queue_pkts': 3, 'source': 6, 'target': 3, 'used_bw': 9.476570678049745}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 2.9087066650390625, 'distance': 14.14, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 334.9997652811736, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 7, 'target': 4, 'used_bw': 0.00023471882640587702}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 3.0362606048583984, 'distance': 10.0, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 414.9998435207824, 'loss': 0.0001427504572832461, 'pkt_drop': 1, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 7, 'target': 10, 'used_bw': 0.00015647921760390641}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 2.9321908950805664, 'distance': 14.14, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 200.9996879570941, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 7, 'target': 9, 'used_bw': 0.00031204290589954863}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 4.267096519470215, 'distance': 10.0, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 387.9997652811736, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 7, 'target': 3, 'used_bw': 0.00023471882640585962}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 3.171205520629883, 'distance': 10.0, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 446.9996881091618, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 4, 'target': 8, 'used_bw': 0.00031189083820663024}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 2.1550655364990234, 'distance': 14.14, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 493.99976608187137, 'loss': 0.11195283054073217, 'pkt_drop': 1, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 8, 'target': 10, 'used_bw': 0.0002339181286549727}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 1.7510652542114258, 'distance': 20.0, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 428.9997031998763, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 8, 'target': 14, 'used_bw': 0.00029680012366673674}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 2.167820930480957, 'distance': 10.0, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 384.9997012603081, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 10, 'target': 13, 'used_bw': 0.00029873969192468645}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 3.4619569778442383, 'distance': 14.14, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 368.9997031998763, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 10, 'target': 14, 'used_bw': 0.00029680012366671587}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 2.9032230377197266, 'distance': 14.14, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 230.9996926524732, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 10, 'target': 12, 'used_bw': 0.0003073475268128555}, {'(3, 15)': 1, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 2.8923749923706055, 'distance': 10.0, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 99.99977781206604, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 11, 'target': 15, 'used_bw': 0.0002221879339608165}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 1.7424821853637695, 'distance': 14.14, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 252.99969201154957, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 11, 'target': 9, 'used_bw': 0.0003079884504331125}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 2.8249025344848633, 'distance': 14.14, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 308.99970374942137, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 15, 'target': 12, 'used_bw': 0.0002962505786144012}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 2.9670000076293945, 'distance': 10.0, 'forward_queue_bytes': 4542, 'forward_queue_pkts': 3, 'free_bw': 426.532551304626, 'loss': 0.03592326095286571, 'pkt_drop': 356, 'pkt_err': 0.0, 'reverse_queue_bytes': 132, 'reverse_queue_pkts': 2, 'source': 16, 'target': 12, 'used_bw': 9.467448695374015}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 4.046797752380371, 'distance': 14.14, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 488.99970429693514, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 16, 'target': 13, 'used_bw': 0.0002957030648390603}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 0, 'delay': 3.22568416595459, 'distance': 10.0, 'forward_queue_bytes': 3028, 'forward_queue_pkts': 2, 'free_bw': 394.53263582519656, 'loss': 0.02377565250865562, 'pkt_drop': 245, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 12, 'target': 9, 'used_bw': 9.467364174803455}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 1, '(3, 18)': 0, 'delay': 4.381775856018066, 'distance': 10.0, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 231.99970352069178, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 13, 'target': 17, 'used_bw': 0.00029647930821494233}, {'(3, 15)': 0, '(3, 16)': 0, '(3, 17)': 0, '(3, 18)': 1, 'delay': 1.6913414001464844, 'distance': 10.0, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 349.9997031998763, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 14, 'target': 18, 'used_bw': 0.00029680012366671587}], 'nodes': [{'id': 5}, {'id': 1}, {'id': 6}, {'id': 2}, {'id': 7}, {'id': 4}, {'id': 8}, {'id': 10}, {'id': 11}, {'id': 15}, {'id': 16}, {'id': 12}, {'id': 13}, {'id': 14}, {'id': 18}, {'id': 17}, {'id': 9}, {'id': 3}], 'timestamp': '2025-07-02-11-05-54'}
    sample_data ={'action': {'(1, 5)': 0.5, '(1, 6)': 0.5}, 'choice_key': '(1, 6)', 'edges': [{'(1, 5)': 0, '(1, 6)': 0, 'delay': 38.251280784606934, 'distance': 1.0, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 0.37745868945869, 'loss': 0.28568801090034046, 'pkt_drop': 18, 'pkt_err': 0.0, 'reverse_queue_bytes': 63588, 'reverse_queue_pkts': 22, 'source': 1, 'target': 3, 'used_bw': 9.62254131054131}, {'(1, 5)': 0, '(1, 6)': 0, 'delay': 7.857799530029297, 'distance': 1.0, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 6.727396771130104, 'loss': 0.020691010790362126, 'pkt_drop': 1, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 1, 'target': 2, 'used_bw': 3.272603228869895}, {'(1, 5)': 0, '(1, 6)': 0, 'delay': 0.981450080871582, 'distance': 1.0, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 9.998474453941121, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 1, 'target': 4, 'used_bw': 0.0015255460588793942}, {'(1, 5)': 0, '(1, 6)': 0, 'delay': 0.8023977279663086, 'distance': 1.0, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 0.3745732289478614, 'loss': 0.1509906174394163, 'pkt_drop': 10, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 3, 'target': 6, 'used_bw': 9.625426771052139}, {'(1, 5)': 0, '(1, 6)': 0, 'delay': 7.4634552001953125, 'distance': 1.0, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 9.998664884475845, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 6, 'target': 4, 'used_bw': 0.00133511552415505}, {'(1, 5)': 1, '(1, 6)': 0, 'delay': 0, 'distance': 1.0, 'forward_queue_bytes': 0, 'forward_queue_pkts': 0, 'free_bw': 6.978282654427236, 'loss': 0.0, 'pkt_drop': 0, 'pkt_err': 0.0, 'reverse_queue_bytes': 0, 'reverse_queue_pkts': 0, 'source': 2, 'target': 5, 'used_bw': 3.0217173455727635}], 'nodes': [{'id': 1}, {'id': 3}, {'id': 6}, {'id': 2}, {'id': 5}, {'id': 4}], 'timestamp': '2025-07-17-10-19-17'}
    for src_dst in setting.src_dst_pair:
        for e in sample_data['edges']:
            # print(f'{src_dst}路径值{e[src_dst]}')
            if e[src_dst] == 1:
                print(e)



    state = State(sample_data)
    G = state.get_graph()
    # # print(G.nodes)
    # # print(G.edges)
    # # action_pro = get_high_action_pro(sample_data)
    # # print(sum(action_pro))
    # # print(state.get_features())
    # # print(f"节点序：{state.nodes_order_tensor}")
    # #
    # get_high_reward_list(sample_data)
    # #
    # # high_reward = get_high_reward(sample_data, action_pro)
    # # print(high_reward)
    # #
    # # low_state, action, low_step_reward, next_low_state = state.get_low_state_action_reward(2, 15, 6)
    # #
    # # # 继续获取下层下一个状态
    # # print(low_state.graph.nodes(data=True))
    # # print(next_low_state.graph.nodes(data=True))
    # #
    # # low_sample = state.get_low_sample()
    # # print(low_sample)
    #
    print(dict(nx.all_pairs_dijkstra_path_length(G,weight='distance')))
    print(dict(nx.all_pairs_shortest_path(G)))