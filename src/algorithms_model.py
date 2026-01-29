#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2025/4/13 10:11
@File:algorithms_model.py
@Desc: 算法设计
"""
import copy

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import setting
import threading
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
LOCK = threading.Lock()

class EVGraphConvolution(nn.Module):
    """
    简单GCN层，类似https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features_v, out_features_v, in_features_e, out_features_e, bias=True, node_layer=True):
        """
        GCN层
        :param in_features_v: 输入节点特征数
        :param out_features_v: 输出节点特征数
        :param in_features_e: 输入边的特征数
        :param out_features_e: 输出边的特征数
        :param bias: 偏置量
        :param node_layer: 是否节点层
        """
        super(EVGraphConvolution, self).__init__()
        self.in_features_v = in_features_v
        self.out_features_v = out_features_v
        self.in_features_e = in_features_e
        self.out_features_e = out_features_e
        self.node_layer = False
        self.have_features_v = True
        # 分为两种情况，是否含有节点特征，若含有又分为是否是节点层
        if self.in_features_v == 0: # 没有节点特征，按边特征卷积处理
            print('不存在节点特征，只做边卷积')
            self.have_features_v = False
            self.weight = nn.Parameter(torch.FloatTensor(in_features_e,out_features_e))
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(out_features_e))
            else:
                self.register_parameter('bias',None)
        else:
            if node_layer:  # 有节点的情况判断是否是节点层,节点层的参数以节点为准
                self.node_layer = True
                self.weight = nn.Parameter(torch.FloatTensor(in_features_v,out_features_v).float())
                self.p = nn.Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_e))).float())
                if bias:
                    self.bias = nn.Parameter(torch.FloatTensor(out_features_v))
                else:
                    self.register_parameter('bias',None)
            else:    # 非节点层，边层
                self.node_layer = False
                self.weight = nn.Parameter(torch.FloatTensor(in_features_e,out_features_e))
                self.p = nn.Parameter(torch.from_numpy(np.random.normal(size=(1,in_features_v))).float())
                if bias:
                    self.bias = nn.Parameter(torch.FloatTensor(out_features_e).float())
                else:
                    self.register_parameter('bias',None)  # 在模块中显式声明一个名为“bias”的参数占位符

        self.reset_parameter()  # 参数重置

    def reset_parameter(self):
        """
        参数重置方法
        :return:
        """
        stdv = 1./ math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv).float()

    def forward(self,H_v,H_e,adj_v,adj_e,T):
        """
        前向传播函数
        :param H_v: 节点特征
        :param H_e: 边特征
        :param adj_v: 节点邻接矩阵
        :param adj_e: 边邻接矩阵
        :param T: 转移矩阵
        :return:
        """
        if self.have_features_v == False: # 没有节点特征的情况
            # H_e^{(l+1)} = \sigma(\widetilde{A}_v H_v^{(v)} W_v)
            # print('self.weight:',self.weight.requires_grad)
            # print(H_e)
            # print(self.weight)
            # weight = self.weight.repeat(H_e.size(0),1,1)
            # print(f'shape:He{H_e.shape},self.weight{self.weight.shape},adj_e{adj_e.shape}')
            output = torch.matmul(adj_e,torch.matmul(H_e,self.weight))
            # print('乘法',torch.matmul(H_e,self.weight).requires_grad)
            if self.bias is not None:
                ret = output + self.bias
            else:
                ret = output
            return H_v, ret  # 节点信息，边信息

        elif self.node_layer:  # 节点层
            multiplier1 = torch.matmul(T,torch.diag_embed((H_e @ self.p.t()).squeeze(-1))) @ T.transpose(1,2)
            mask1 = torch.eye(multiplier1.shape[1]).to(setting.device).float()
            M1 = mask1 * torch.ones(multiplier1.shape[1]).to(setting.device).float() + (1. - mask1)*multiplier1
            adjusted_A = torch.mul(M1,adj_v)
            """
            为了避免丢失特征的信息，我们这里不归一化A
            """
            output = torch.matmul(adjusted_A,torch.matmul(H_v,self.weight))
            if self.bias is not None:
                ret = output + self.bias
            else:
                ret = output
            return ret, H_e  # 节点信息，边信息

        else:  # 边层
            multiplier2 = torch.matmul(T.transpose(1,2),torch.diag_embed((H_v @ self.p.t()).squeeze(-1))) @ T
            mask2 = torch.eye(multiplier2.shape[1]).to(setting.device).float()
            M2 = mask2 * torch.ones(multiplier2.shape[1]).to(setting.device).float() + (1. - mask2)*multiplier2
            adjusted_A = torch.mul(M2,adj_e)
            normalized_adjusted_A = adjusted_A/adjusted_A.max(1,keepdim=True)[0]
            output = torch.matmul(normalized_adjusted_A,torch.matmul(H_e,self.weight))
            if self.bias is not None:
                ret = output + self.bias
            else:
                ret = output
            return H_v, ret  # 节点信息，边信息

    def __repr__(self):  # 打印信息
        return self.__class__.__name__ + ' (' \
               + str(self.in_features_v) + ' -> ' \
               + str(self.out_features_v) + ')'+'\n' + ' (' \
               + str(self.in_features_e) + ' -> ' \
               + str(self.out_features_e) + ')'

class HighActor(nn.Module):
    """
    上层动作网络，没有节点特征
    """
    def __init__(self,nfeat_e,nhid,nout,dropout):
        """
        :param nfeat_e: 边的特征数
        :param nhid: 隐藏层输入-输出维度
        :param nout: 输出维度
        :param dropout: 丢弃率
        """
        super(HighActor, self).__init__()
        self.gc1 = EVGraphConvolution(0,0,nfeat_e,nhid,node_layer=False)
        self.gc2 = EVGraphConvolution(0,0,nhid,nhid,node_layer=False)
        self.lin1 = nn.Linear(nhid,2*nhid)
        self.lin2 = nn.Linear(2*nhid,nout)
        self.dropout = dropout

    def forward(self,_state):
        """
        前向传播 H_v,H_e,adj_v,adj_e,T
        :param H_e: 边特征
        :param adj_e: 边邻接矩阵
        :return:
        """
        H_e = _state.H_e
        adj_e = _state.adj_e
        # print('输入H_e',H_e)
        # print('输入adj_e',adj_e)
        # 第1层
        X, Z = self.gc1(None,H_e,None,adj_e,None)
        # print('Z0', Z.requires_grad)
        Z = F.leaky_relu(Z)
        # print('Z1', Z.requires_grad)
        Z = F.dropout(Z, self.dropout, training=self.training)

        # print('中间结果-1', Z)
        # 第2层
        X, Z = self.gc2(None, Z, None, adj_e, None)
        Z =  F.leaky_relu(Z)
        # print('Z2', Z.requires_grad)
        Z =  F.dropout(Z, self.dropout, training=self.training)
        # print('中间结果0', Z)
        # 中间层
        Z = torch.sum(Z,dim=1)
        # print('中间结果1', Z)
        # 线性层
        Z = F.leaky_relu(self.lin1(Z))
        Z = F.dropout(Z, self.dropout, training=self.training)
        Z = F.leaky_relu(self.lin2(Z))
        Z = F.dropout(Z, self.dropout, training=self.training)
        # print('中间结果3',Z)

        Z = torch.clamp(Z, min=-10.0, max=10.0)
        Z = F.gumbel_softmax(Z, hard=False, tau=0.5)
        # print('Z',Z.requires_grad)
        return Z


class HighCritic(nn.Module):
    """
    上层评价网络，带节点特征但主要是边的特征
    """
    def __init__(self, nfeat_v, nfeat_e, nhid, dropout):
        """
        :param nfeat_v: 节点特征数
        :param nfeat_e: 边特征数
        :param nhid: 隐藏层输入-输出维度
        :param dropout: 参数丢弃率
        """
        super(HighCritic, self).__init__()
        self.gc1 = EVGraphConvolution(nfeat_v, nfeat_v, nfeat_e, nhid, node_layer=False)
        self.gc2 = EVGraphConvolution(nfeat_v, nfeat_v, nhid, nhid, node_layer=False)
        self.lin1 = nn.Linear(nhid, 2*nhid)
        self.lin2 = nn.Linear(2*nhid, 1)
        self.dropout = dropout

    def forward(self, _state):
        """
        前向传播
        :param H_v: 节点特征
        :param H_e:  边特征
        :param adj_v: 节点邻接矩阵
        :param adj_e: 边邻接矩阵
        :param T: 节点-边转换矩阵
        :return: 动作概率
        """
        try:
            H_v = _state.H_v
            H_e = _state.H_e
            adj_v = _state.adj_v
            adj_e = _state.adj_e
            T = _state.T
        except Exception as e:
            # print(f'[模型]{e}，没有节点特征，只做边卷积')
            H_e = _state.H_e
            adj_e = _state.adj_e
            # 第1层
            X, Z = self.gc1(None, H_e, None, adj_e, None)
            Z = F.leaky_relu(Z)
            Z = F.dropout(Z, self.dropout, training=self.training)

            # print('中间结果-1', Z)
            # 第2层
            X, Z = self.gc2(None, Z, None, adj_e, None)
            Z = F.leaky_relu(Z)
            Z = F.dropout(Z, self.dropout, training=self.training)
            # print('中间结果0', Z)
            # 中间层
            Z = torch.sum(Z, dim=1)
            # print('中间结果1', Z)
            # 线性层
            Z = F.leaky_relu(self.lin1(Z))
            Z = F.dropout(Z, self.dropout, training=self.training)
            Z = self.lin2(Z)
            return Z

        # 第1层
        X, Z = self.gc1(H_v, H_e, adj_v, adj_e, T)
        Z = F.leaky_relu(Z)
        Z = F.dropout(Z, self.dropout, training=self.training)
        # 第2层
        X, Z = self.gc2(X, Z, adj_v, adj_e, T)
        Z = F.leaky_relu(Z)
        Z = F.dropout(Z, self.dropout, training=self.training)

        # 中间层
        Z = torch.sum(Z, dim=1)

        # 线性层
        Z = F.leaky_relu(self.lin1(Z))
        Z = F.dropout(Z, self.dropout, training=self.training)
        Z = self.lin2(Z)
        return Z

class LowActor(nn.Module):
    """
    下层动作网络，存在两列节点层，当前位置和目标位置
    """
    def __init__(self,nfeat_e,nfeat_v,nhid,dropout):
        """
        下层动作网络
        :param nfeat_e: 边特征数
        :param nfeat_v: 节点特征数
        :param nhid: 隐藏层输入-输出维度
        :param dropout: 参数丢弃率
        """
        super(LowActor, self).__init__()
        self.gc1 = EVGraphConvolution(nfeat_v,1,nfeat_e,nhid,node_layer=False)
        self.gc2 = EVGraphConvolution(nfeat_v,1,nhid,nhid,node_layer=False)
        self.gc3 = EVGraphConvolution(nfeat_v,1,nhid,nhid,node_layer=True)
        self.dropout = dropout


    def forward(self,_state,evaluate=False):
        """
        前向传播
        :param H_v: 节点特征
        :param H_e: 边特征
        :param adj_v: 节点邻接矩阵
        :param adj_e: 边邻接矩阵
        :param T: 节点-边转移矩阵
        :return:
        """
        with LOCK:
            # print(f"_state id{id(_state)}")
            H_v = _state.H_v
            H_e = _state.H_e
            adj_v = _state.adj_v
            adj_e = _state.adj_e
            T = _state.T
            Mask = _state.Mask

            # 第一层
            X, Z  = self.gc1(H_v,H_e,adj_v,adj_e,T)
            Z = F.leaky_relu(Z)
            Z = F.dropout(Z,self.dropout,training=self.training)

            # 第二层
            X, Z = self.gc2(X, Z, adj_v, adj_e, T)
            Z = F.leaky_relu(Z)
            Z = F.dropout(Z, self.dropout, training=self.training)

            # 第三层
            X, Z = self.gc3(X, Z, adj_v, adj_e, T)

            # 取各自的邻居节点（只能在批量的情况执行）
            adj_v_ = torch.mul(1.-torch.eye(adj_v.size()[1]).to(setting.device),adj_v.detach())
            nei_nzero = torch.matmul(H_v[:, :, 0].detach().unsqueeze(1), adj_v_)
            nei_mask = (nei_nzero>0).float()  # 邻居掩码
            if evaluate:
                total_mask = nei_mask
            else:
                total_mask = torch.mul(nei_mask, Mask.unsqueeze(1))
            assert torch.all(torch.sum(total_mask,dim=2)>0) == True, f'没有下一跳,特征向量{H_v},邻接矩阵{adj_v},邻居{nei_mask}和序掩码{Mask},状态id{id(_state)}'

            X = torch.matmul(X.transpose(1,2),torch.mul(total_mask.transpose(1,2),torch.eye(adj_v.size()[1]).to(setting.device)))  # 哈达玛积
            X = X.masked_fill(X==0,-1e9)  # 不希望在X==0的地方取到1，因为这些地方本来就进行了处理，不需要梯度传递 (只使用这一句就行，这样可能会出问题。)
            X = F.gumbel_softmax(X,tau=1,hard=True)
        return X

class LowCritic(nn.Module):
    """
    下层评论网络，节点特征当前位置，目标位置，动作选择
    """
    def __init__(self,nfeat_v,nfeat_e,nhid,dropout):
        """
        :param nfeat_v: 节点特征数
        :param nfeat_e: 边特征数
        :param nhid: 隐藏层输入-输出
        :param dropout: 参数丢弃率
        """
        super(LowCritic, self).__init__()
        self.gc1 = EVGraphConvolution(nfeat_v,nfeat_v,nfeat_e,nhid,node_layer=False)
        self.gc2 = EVGraphConvolution(nfeat_v,nfeat_v,nhid,nhid,node_layer=False)
        self.lin1 = nn.Linear(nhid,2*nhid)
        self.lin2 = nn.Linear(2*nhid,1)
        self.dropout = dropout

    def forward(self, _state):
        """
        前向传播过程
        :param H_v: 节点特征
        :param H_e: 边特征
        :param adj_v: 节点邻接矩阵
        :param adj_e: 边邻接矩阵
        :param T: 节点-边转移矩阵
        :return:
        """
        H_v = _state.H_v
        H_e = _state.H_e
        adj_v = _state.adj_v
        adj_e = _state.adj_e
        T = _state.T

        # 第1层
        X, Z = self.gc1(H_v, H_e, adj_v, adj_e, T)
        Z = F.leaky_relu(Z)
        Z = F.dropout(Z, self.dropout, training=self.training)
        # 第2层
        X, Z = self.gc2(X, Z, adj_v, adj_e, T)
        Z = F.leaky_relu(Z)
        Z = F.dropout(Z, self.dropout, training=self.training)

        # 中间层
        Z = torch.sum(Z, dim=1)

        # 线性层
        Z = F.relu(self.lin1(Z))
        Z = F.dropout(Z, self.dropout, training=self.training)
        Z = self.lin2(Z)
        return Z


class HighDDPG:
    def __init__(self, nfeat_e,nhid, nout,
                 act_lr, cri_lr, dropout, gamma,
                 sigma,tau, device):
        """
        上层DDPG模型
        :param act_input_dim:
        :param act_output_dim:
        :param cri_input_dim:
        :param act_lr:
        :param cri_lr:
        :param gamma:
        :param tau:
        :param device:
        """
        # 策略网络
        self.actor = HighActor(nfeat_e, nhid, nout, dropout).to(device)
        # 目标策略网络
        self.target_actor = HighActor(nfeat_e, nhid, nout, dropout).to(device)
        # 评价网络
        self.critic = HighCritic(1,nfeat_e,nhid,dropout).to(device)
        # 目标评价网络
        self.target_critic = HighCritic(1,nfeat_e,nhid,dropout).to(device)

        self.device = device  # 设备
        self.gamma = gamma  # 折扣率
        self.tau = tau  # 软更新率
        self.critic_loss = torch.nn.MSELoss() # 评价方法
        self.sigma = sigma  # 高斯噪声的标准差，均值直接设为0
        self.act_dim = nout  # 动作的维度

        # 目标参数初始化
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 定义迭代器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=act_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cri_lr)

    def take_action(self,state):  # 用于外界调用，应用模型
        model = copy.deepcopy(self.actor)
        model.eval()
        act = model(state).item()   # 获取动作
        # 添加噪声，增加探索率
        act = act + self.sigma * np.random.randn(self.act_dim)
        return act


    def soft_update(self,net,target_net):
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data * (1. - self.tau) +
            param.data * self.tau)


    def update(self,samples):
        # sample: units×batch
        state_list = []
        act_list = []
        rew_list = []
        next_state_list = []
        for sample in samples:

            state, act, rew, next_state = sample[0]
            state_list.append(state)
            act_list.append(act)
            rew_list.append(rew)
            next_state_list.append(next_state)

        _next_act_input = gen_high_act_input_batch(next_state_list).to(self.device)  # 下一个状态
        _next_act = self.target_actor(_next_act_input)
        # print('动作', _next_act)

        _next_cri_input = gen_high_cri_input_batch(next_state_list, _next_act).to(self.device)
        # _next_cri = self.critic(_next_cri_input)
        # print('Q-值', _next_cri)

        next_q_values = self.target_critic(_next_cri_input)
        # 下一个状态，从目标actor网络获得的下一个状态动作,通过目标critic网络获得
        q_targets = torch.tensor(rew_list).float().to(setting.device).unsqueeze(1) + self.gamma * next_q_values

        # 真实Q值
        q_values = self.critic(gen_high_cri_input_batch(state_list, act_list).to(setting.device))
        # 当前状态和采样动作

        # Loss
        critic_loss = torch.mean(F.mse_loss(q_values, q_targets))

        # 方向传播
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor Loss
        actor_loss = -torch.mean(self.critic(gen_high_cri_input_batch(state_list, self.actor(gen_high_act_input_batch(state_list).to(setting.device))).to(setting.device)))
        # 当前状态，以及从actor网络获得的动作
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新过程
        self.soft_update(self.actor,self.target_actor)
        self.soft_update(self.critic,self.target_critic)



def gen_high_act_input_batch(state_list):
    """
    生成策略网络的输出(Data(H_e,H_v,adj_e,adj_v,T))
    :param state: 状态
    :return:
    """

    data_list = [
        Data(
            H_e = state.torch_norm_efeatures.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_norm_efeatures is not None else None,
            H_v = state.torch_norm_features.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_norm_features is not None else None,
            adj_e = state.torch_norm_eadj.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_norm_eadj is not None else None,
            adj_v = state.torch_norm_adj.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_norm_adj is not None else None,
            T = state.torch_trans_mx.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_trans_mx is not None else None,
            # node_orders = state.graph.nodes()
        )
        for state in state_list]

    try:
        batch = Batch.from_data_list(data_list)
    except RuntimeError as e:
        print('state_list',state_list)
        print(f'错误{e}发生')
        return
    # print("策略网络输入完成！")
    return batch

def gen_low_act_input_batch(state_list):
    """
    生成策略网络的输出(Data(H_e,H_v,adj_e,adj_v,T))
    :param state: 状态
    :return:
    """
    data_list = [
        Data(
            H_e = state.torch_norm_efeatures.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_norm_efeatures is not None else None,
            H_v = state.torch_norm_features.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_norm_features is not None else None,
            adj_e = state.torch_norm_eadj.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_norm_eadj is not None else None,
            adj_v = state.torch_norm_adj.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_norm_adj is not None else None,
            T = state.torch_trans_mx.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_trans_mx is not None else None,
            Mask = state.nodes_order_tensor.float().unsqueeze(0).to(setting.device) if state.nodes_order_tensor is not None else None,
            # node_orders = state.graph.nodes()
        )
        for state in state_list]

    try:
        batch = Batch.from_data_list(data_list)
    except RuntimeError as e:
        print('state_list', state_list)
        print(f'错误{e}发生')
        return
        # print("策略网络输入完成！")
    return batch



def sparse_unsqueeze(sparse_tensor,dim):
    """
    稀疏矩阵的升维
    :param sparse_tensor:
    :param dim:
    :return:
    """
    # 1. 强制合并张量
    coaleced_tensor = sparse_tensor.coalesce()

    # 2. 获取合并和的元数据
    indices = coaleced_tensor.indices()
    values = coaleced_tensor.values()

    # 正确获取数据类型和设备信息
    dtype = indices.dtype  # 正确方式：使用.dtype属性
    device = indices.device  # 推荐使用张量自带的设备信息
    original_sparse_dim = sparse_tensor.sparse_dim()
    new_sparse_dim = original_sparse_dim+1

    # 3. 计算非零元素数
    nnz = indices.size(1)

    # 4. 构造新索引
    new_indices = torch.cat([
        torch.zeros(1,nnz,dtype=dtype,device=device),
        indices
    ], dim=dim)

    # 5.调整维度
    new_shape = list(coaleced_tensor.shape)
    new_shape.insert(dim, 1)

    # 6.构建新张量并再次合并
    return torch.sparse_coo_tensor(
        indices = new_indices,
        values = values,
        size = new_shape,
        sparse_dim = new_sparse_dim,  # 关键修复点
        dense_dim = coaleced_tensor.dense_dim()
    ).coalesce()




def gen_high_cri_input_batch(state_list, act):
    """

    :param state_list:
    :param act:
    :return:
    """
    nodes = list(state_list[0].graph.nodes())  # 获取节点信息
    node_len = len(nodes)  # 获取节点个数
    dst_num = act.shape[1] if isinstance(act,torch.Tensor) else len(act[0])
    matrix = torch.zeros((dst_num, node_len), dtype=torch.float, requires_grad=False).to(setting.device)
    dst_list = setting.dst_list
    for row, dst in zip(range(dst_num),dst_list):
        col = nodes.index(dst)
        matrix[row, col] = 1  # 将概率转化到相应的目的节点所在的序列，并保证梯度可回传

    # print(matrix)
    if isinstance(act,torch.Tensor):
        H_v_batch = torch.matmul(act, matrix).unsqueeze(1).transpose(1,2)
    elif isinstance(act,list):
        act_tensor = torch.tensor(act,requires_grad=False).float().to(setting.device)
        H_v_batch = torch.matmul(act_tensor, matrix).unsqueeze(1).transpose(1, 2)
    # print(H_v_batch)
    else:
        print(f"[ERROR]暂时不支持这种类型的数据{type(act)}，请检查。。。")
        H_v_batch = None


    data_list = [
        Data(
            H_e=state.torch_norm_efeatures.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_norm_efeatures is not None else None,
            H_v=_H_v.unsqueeze(0).float().to(setting.device) if _H_v is not None else None,
            adj_e=state.torch_norm_eadj.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_norm_eadj is not None else None,
            adj_v=state.torch_norm_adj.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_norm_adj is not None else None,
            T=state.torch_trans_mx.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_trans_mx is not None else None
        )
        for state,_H_v in zip(state_list,H_v_batch)]

    try:
        batch = Batch.from_data_list(data_list)
    except RuntimeError as e:
        print('state_list', state_list)
        print(f'错误{e}发生')

        return
        # print("评价网络输入完成！")
    return batch

class LowDDPG:
    def __init__(self, nfeat_e,nfeat_v,nhid,
                 act_lr, cri_lr, dropout, gamma,
                 sigma,tau, device):
        """
        下层DDPG模型
        :param act_input_dim:
        :param act_output_dim:
        :param cri_input_dim:
        :param act_lr:
        :param cri_lr:
        :param gamma:
        :param tau:
        :param device:
        """
        # 策略网络
        self.actor = LowActor(nfeat_e, nfeat_v, nhid, dropout).to(device)
        # 目标策略网络
        self.target_actor = LowActor(nfeat_e, nfeat_v, nhid, dropout).to(device)
        # 评价网络
        self.critic = LowCritic(nfeat_v+1, nfeat_e, nhid, dropout).to(device)
        # 目标评价网络
        self.target_critic = LowCritic(nfeat_v+1, nfeat_e, nhid, dropout).to(device)

        self.device = device  # 设备
        self.gamma = gamma  # 折扣率
        self.tau = tau  # 软更新率
        self.critic_loss = torch.nn.MSELoss()  # 评价方法
        self.sigma = sigma  # 高斯噪声的标准差，均值直接设为0
        self.act_dim = 1  # 动作的维度

        # 目标参数初始化
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 定义迭代器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=act_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cri_lr)

    def take_action(self,state):  # 用于外界调用，应用模型
        model = copy.deepcopy(self.actor)
        model.eval()
        act = model(state).item()   # 获取动作
        # 添加噪声，增加探索率
        act = act + self.sigma * np.random.randn(self.act_dim)
        return act


    def soft_update(self,net,target_net):
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data * (1. - self.tau) +
            param.data * self.tau)


    def update(self,samples):
        # sample: units×batch
        state_list = []
        act_list = []
        rew_list = []
        next_state_list = []
        done_list = []
        for sample in samples:
            state, act, rew, next_state, done = sample[0]
            state_list.append(state)
            act_list.append(act)
            rew_list.append(rew)
            next_state_list.append(next_state)
            done_list.append(done)

        _next_act_input = gen_low_act_input_batch(next_state_list).to(self.device)  # 下一个状态
        _next_act = self.target_actor(_next_act_input)
        # print('动作', _next_act)

        _next_cri_input = gen_low_cri_input_batch(next_state_list, _next_act).to(self.device)
        # _next_cri = self.critic(_next_cri_input)
        # print('Q-值', _next_cri)

        next_q_values = self.target_critic(_next_cri_input)
        # 下一个状态，从目标actor网络获得的下一个状态动作,通过目标critic网络获得
        q_targets = torch.tensor(rew_list).float().to(setting.device).unsqueeze(1) + self.gamma * next_q_values

        # 真实Q值
        q_values = self.critic(gen_low_cri_input_batch(state_list, act_list).to(self.device))
        # 当前状态和采样动作

        # Loss
        critic_loss = torch.mean(F.mse_loss(q_values, q_targets))

        # 方向传播
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor Loss
        actor_loss = -torch.mean(self.critic(gen_low_cri_input_batch(state_list, self.actor(gen_low_act_input_batch(state_list).to(setting.device))).to(setting.device)))
        # 当前状态，以及从actor网络获得的动作
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新过程
        self.soft_update(self.actor,self.target_actor)
        self.soft_update(self.critic,self.target_critic)


def gen_low_cri_input_batch(state_list, act):
    """
    获取下层评价网络的输入
    :param state_list:
    :param act:
    :return:
    """

    if isinstance(act,list):
        nodes = list(state_list[0].graph.nodes())   # 获取节点信息
        node_len = len(nodes)
        batch = len(act)
        H_v_batch = torch.zeros((batch,1,node_len),dtype=torch.float,requires_grad=False).to(setting.device)
        for dim1,act_dict in enumerate(act):
            for dim3,node in enumerate(nodes):
                H_v_value = act_dict[node]
                H_v_batch[dim1,0,dim3] = H_v_value
    elif isinstance(act,torch.Tensor):
        H_v_batch = act
    else:
        print(f"[ERROR]暂时不支持这种类型的数据{type(act)}，请检查。。。")
        H_v_batch = None


    data_list = [
        Data(
            H_e=state.torch_norm_efeatures.to_dense().float().unsqueeze(
                0).to(setting.device) if state.torch_norm_efeatures is not None else None,
            H_v=torch.cat([state.torch_norm_features.to_dense().float().unsqueeze(0).to(setting.device),_H_v.unsqueeze(0).transpose(1, 2).float()],dim=2) if _H_v is not None else None,
            adj_e=state.torch_norm_eadj.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_norm_eadj is not None else None,
            adj_v=state.torch_norm_adj.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_norm_adj is not None else None,
            T=state.torch_trans_mx.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_trans_mx is not None else None
        )
        for state, _H_v in zip(state_list, H_v_batch)]


    batch = Batch.from_data_list(data_list)
    # print("评价网络输入完成！")
    return batch

class HighPPO:
    def __init__(self, nfeat_e,nhid, nout, dropout,
                 act_lr, cri_lr,
                 lmbda, epochs, eps,
                 gamma, device, entropy_coef=0.01):
        """
        上层PPO模型
        :param act_lr:
        :param cri_lr:
        :param gamma:
        :param device:
        """
        # 策略网络
        self.actor = HighActor(nfeat_e, nhid, nout, dropout).to(device)

        # 评价网络
        self.critic = HighCritic(0,nfeat_e,nhid,dropout).to(device)

        self.device = device  # 设备
        self.gamma = gamma  # 折扣率
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列用来训练轮数
        self.act_dim = nout  # 动作的维度
        self.eps = eps
        self.entropy_coef = entropy_coef

        # 定义迭代器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=act_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cri_lr)

    def take_action(self,state):  # 用于外界调用，应用模型
        model = copy.deepcopy(self.actor)
        model.eval()
        act = model(state).item()   # 获取动作
        return act

    def update(self,samples):
        # sample: units×batch
        state_list = []
        act_list = []
        rew_list = []
        next_state_list = []
        for sample in samples:
            state, act, rew, next_state = sample[0]
            state_list.append(state)
            act_list.append(act)
            rew_list.append(rew)
            next_state_list.append(next_state)


        _next_cri_input = gen_high_ppo_cri_input_batch(next_state_list).to(self.device)  # 下一个状态作为输入
        _net_cri = self.critic(_next_cri_input)

        td_target = torch.from_numpy(np.array(rew_list)).float().unsqueeze(1).to(self.device) + self.gamma * _net_cri

        _cri_input = gen_high_ppo_cri_input_batch(state_list).to(self.device)
        _cri = self.critic(_cri_input)

        td_delta = td_target - _cri

        advantage = compute_advantage(self.gamma,self.lmbda,td_delta.cpu()).to(self.device)

        _act_input = gen_high_act_input_batch(state_list).to(self.device)
        old_log_pros = torch.log(self.actor(_act_input).gather(1,torch.tensor(act_list).unsqueeze(1).to(self.device))).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(_act_input).gather(1,torch.tensor(act_list).unsqueeze(1).view(-1,1).to(self.device)))
            # print(f'log_probs:{log_probs.requires_grad}')
            ratio = torch.exp(log_probs-old_log_pros)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage  # 截断
            # 计算熵
            entropy = - (torch.exp(log_probs)*log_probs).sum(dim=1).mean()

            actor_loss = torch.mean(-torch.min(surr1,surr2)) - self.entropy_coef * entropy  # PPO损失函数
            critic_loss = torch.mean(F.mse_loss(self.critic(gen_high_ppo_cri_input_batch(state_list).to(self.device)),td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # (f'actor_loss:{actor_loss.requires_grad}')
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


def gen_high_ppo_cri_input_batch(state_list):
    """
    生成策略网络的输出(Data(H_e,H_v,adj_e,adj_v,T))
    :param state_list: 状态
    :return:
    """
    data_list = [
        Data(
            H_e = state.torch_norm_efeatures.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_norm_efeatures is not None else None,
            H_v = state.torch_norm_features.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_norm_features is not None else None,
            adj_e = state.torch_norm_eadj.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_norm_eadj is not None else None,
            adj_v = state.torch_norm_adj.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_norm_adj is not None else None,
            T = state.torch_trans_mx.to_dense().float().unsqueeze(0).to(setting.device) if state.torch_trans_mx is not None else None,
        )
        for state in state_list]

    batch = Batch.from_data_list(data_list)
    # print("策略网络输入完成！")
    return batch

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

## SAC算法
class HighSACActor(nn.Module):
    """
    上层动作网络，没有节点特征
    """
    def __init__(self,nfeat_e,nhid,nout,dropout,temperature=setting.high_temperature):
        """
        :param nfeat_e: 边的特征数
        :param nhid: 隐藏层输入-输出维度
        :param nout: 输出维度
        :param dropout: 丢弃率
        """
        super(HighSACActor, self).__init__()
        self.gc1 = EVGraphConvolution(0,0,nfeat_e,nhid,node_layer=False)
        self.gc2 = EVGraphConvolution(0,0,nhid,nhid,node_layer=False)
        self.lin1 = nn.Linear(nhid,2*nhid)
        self.lin2 = nn.Linear(2*nhid,nout)
        self.dropout = dropout
        self.temperature = temperature


    def forward(self,_state):
        """
        前向传播 H_v,H_e,adj_v,adj_e,T
        :param H_e: 边特征
        :param adj_e: 边邻接矩阵
        :return:
        """
        H_e = _state.H_e
        adj_e = _state.adj_e
        # print('输入H_e',H_e)
        # print('输入adj_e',adj_e)
        # 第1层
        X, Z = self.gc1(None,H_e,None,adj_e,None)
        # print('Z0', Z.requires_grad)
        Z = F.leaky_relu(Z)
        # print('Z1', Z.requires_grad)
        Z = F.dropout(Z, self.dropout, training=self.training)

        # print('中间结果-1', Z)
        # 第2层
        X, Z = self.gc2(None, Z, None, adj_e, None)
        Z =  F.leaky_relu(Z)
        # print('Z2', Z.requires_grad)
        Z =  F.dropout(Z, self.dropout, training=self.training)
        # print('中间结果0', Z)
        # 中间层
        Z = torch.sum(Z,dim=1)
        # print('中间结果1', Z)
        # 线性层
        Z = F.leaky_relu(self.lin1(Z))
        Z = F.dropout(Z, self.dropout, training=self.training)
        Z = F.leaky_relu(self.lin2(Z))
        Z = F.dropout(Z, self.dropout, training=self.training)
        # print('中间结果3',Z)
        # Z = F.gumbel_softmax(Z, tau=1)  # 进行归一化
        dist = RelaxedOneHotCategorical(self.temperature,logits=Z)
        action = dist.rsample()  # 重参数化采样，输出概率向量
        log_prob = dist.log_prob(action)

        return action,log_prob

class HighSACCritic(nn.Module):
    """
    上层评价网络，带节点特征但主要是边的特征
    """
    def __init__(self, nfeat_v, nfeat_e, nhid, dropout):
        """
        :param nfeat_v: 节点特征数
        :param nfeat_e: 边特征数
        :param nhid: 隐藏层输入-输出维度
        :param dropout: 参数丢弃率
        """
        super(HighSACCritic, self).__init__()
        self.gc1 = EVGraphConvolution(nfeat_v, nfeat_v, nfeat_e, nhid, node_layer=False)
        self.gc2 = EVGraphConvolution(nfeat_v, nfeat_v, nhid, nhid, node_layer=False)
        self.lin1 = nn.Linear(nhid, 2*nhid)
        self.lin2 = nn.Linear(2*nhid, 1)
        self.dropout = dropout

    def forward(self, _state):
        """
        前向传播
        :param H_v: 节点特征
        :param H_e:  边特征
        :param adj_v: 节点邻接矩阵
        :param adj_e: 边邻接矩阵
        :param T: 节点-边转换矩阵
        :return: 动作概率
        """
        try:
            H_v = _state.H_v
            H_e = _state.H_e
            adj_v = _state.adj_v
            adj_e = _state.adj_e
            T = _state.T
        except Exception as e:
            # print(f'[模型]{e}，没有节点特征，只做边卷积')
            H_e = _state.H_e
            adj_e = _state.adj_e
            # 第1层
            X, Z = self.gc1(None, H_e, None, adj_e, None)
            Z = F.leaky_relu(Z)
            Z = F.dropout(Z, self.dropout, training=self.training)

            # print('中间结果-1', Z)
            # 第2层
            X, Z = self.gc2(None, Z, None, adj_e, None)
            Z = F.leaky_relu(Z)
            Z = F.dropout(Z, self.dropout, training=self.training)
            # print('中间结果0', Z)
            # 中间层
            Z = torch.sum(Z, dim=1)
            # print('中间结果1', Z)
            # 线性层
            Z = F.leaky_relu(self.lin1(Z))
            Z = F.dropout(Z, self.dropout, training=self.training)
            Z = self.lin2(Z)
            return Z

        # 第1层
        X, Z = self.gc1(H_v, H_e, adj_v, adj_e, T)
        Z = F.leaky_relu(Z)
        Z = F.dropout(Z, self.dropout, training=self.training)
        # 第2层
        X, Z = self.gc2(X, Z, adj_v, adj_e, T)
        Z = F.leaky_relu(Z)
        Z = F.dropout(Z, self.dropout, training=self.training)

        # 中间层
        Z = torch.sum(Z, dim=1)

        # 线性层
        Z = F.leaky_relu(self.lin1(Z))
        Z = F.dropout(Z, self.dropout, training=self.training)
        Z = self.lin2(Z)
        return Z

class HighSAC:
    def __init__(self, nfeat_e,nhid, nout,
                 act_lr, cri_lr, dropout, alpha_lr, target_entropy, gamma,
                 tau, device):
        """
        上层DDPG模型
        :param act_input_dim:
        :param act_output_dim:
        :param cri_input_dim:
        :param act_lr:
        :param cri_lr:
        :param gamma:
        :param tau:
        :param device:
        """
        # 策略网络
        self.actor = HighSACActor(nfeat_e, nhid, nout, dropout).to(device)

        # 评价网络
        self.critic_1 = HighSACCritic(1,nfeat_e,nhid,dropout).to(device)
        self.critic_2 = HighSACCritic(1,nfeat_e,nhid,dropout).to(device)

        # 目标评价网络
        self.target_critic_1 = HighSACCritic(1,nfeat_e,nhid,dropout).to(device)
        self.target_critic_2 = HighSACCritic(1,nfeat_e,nhid,dropout).to(device)

        # 目标参数初始化
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # 定义迭代器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=act_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=cri_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=cri_lr)

        # 使用alpha的log值，可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01),dtype=torch.float,requires_grad=True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma  # 折扣率
        self.tau = tau  # 软更新率
        self.device = device  # 设备

    def take_action(self,state):  # 用于外界调用，应用模型
        state = torch.tensor([state],dtype=torch.float).to(self.device)
        model = copy.deepcopy(self.actor)
        model.eval()
        _act_input = gen_high_act_input_batch(state)
        act = model(state)[0].item()   # 获取动作
        # 添加噪声，增加探索率
        return act

    def calc_target(self,rewards,next_states):   # 计算Q值，没有终点
        _next_act_input = gen_high_act_input_batch(next_states)
        _next_act,log_prob = self.actor(_next_act_input)
        entropy = -log_prob
        _cri_input = gen_high_cri_input_batch(next_states,_next_act)
        q1_value = self.target_critic_1(_cri_input)
        q2_value = self.target_critic_2(_cri_input)
        next_value = torch.min(q1_value,q2_value) + self.log_alpha.exp() * entropy
        td_target = torch.tensor(rewards,requires_grad=False).to(setting.device).float() + self.gamma * next_value
        return td_target

    def soft_update(self,net,target_net):
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data * (1. - self.tau) +
            param.data * self.tau)


    def update(self,samples):
        # sample: units×batch
        state_list = []
        act_list = []
        rew_list = []
        next_state_list = []
        for sample in samples:
            state, act, rew, next_state = sample[0]
            state_list.append(state)
            act_list.append(act)
            rew_list.append(rew)
            next_state_list.append(next_state)

        # 更新两个Q网络
        td_target = self.calc_target(rew_list, next_state_list)
        _cri_input = gen_high_cri_input_batch(state_list,act_list)
        cri_1_loss = torch.mean(F.mse_loss(self.critic_1(_cri_input),td_target.detach()))
        cri_2_loss = torch.mean(F.mse_loss(self.critic_2(_cri_input),td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        cri_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        cri_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        _act_input = gen_high_act_input_batch(state_list)
        new_actions,log_prob = self.actor(_act_input)
        entropy = -log_prob
        _cri_input = gen_high_cri_input_batch(state_list,new_actions)
        q1_value = self.critic_1(_cri_input)
        q2_value = self.critic_2(_cri_input)
        act_loss = torch.mean(-self.log_alpha.exp()*entropy-torch.min(q1_value,q2_value))
        self.actor_optimizer.zero_grad()
        act_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean((entropy-self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1,self.target_critic_1)
        self.soft_update(self.critic_2,self.target_critic_2)

