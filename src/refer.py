#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2025/3/24 10:33
@File:refer.py
@Desc:大模型参考
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# 自定义数据集生成类
class NetworkDataset(Dataset):
    def __init__(self, num_samples=10000):
        """
        生成合成数据集：
        - bandwidth: 带宽(Mbps) [1, 100]
        - delay: 基础时延(ms) [1, 100]
        - loss: 丢包率 [0, 0.3]
        """
        self.num_samples = num_samples
        np.random.seed(42)

        # 生成原始特征
        bandwidth = np.random.uniform(1, 100, size=num_samples)
        delay = np.random.uniform(1, 100, size=num_samples)
        loss = np.random.uniform(0, 0.3, size=num_samples)

        # 计算理论时延（包含噪声）
        data_size = 1500 * 8  # 1500字节转换为bit
        transmission_delay = data_size / (bandwidth * 1e6) * 1e3  # 转换为毫秒
        e2e_delay = (transmission_delay + delay) / (1 - loss + 1e-6)
        e2e_delay += np.random.normal(0, 0.1, size=num_samples)  # 添加噪声

        # 数据标准化
        self.X = np.column_stack((bandwidth, delay, loss))
        self.X_mean, self.X_std = self.X.mean(axis=0), self.X.std(axis=0)
        self.y_mean, self.y_std = e2e_delay.mean(), e2e_delay.std()

        self.X = (self.X - self.X_mean) / self.X_std
        self.y = (e2e_delay - self.y_mean) / self.y_std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X[idx]),
            torch.FloatTensor([self.y[idx]])
        )


# 神经网络模型类
class E2EDelayModel(nn.Module):
    def __init__(self, input_size=3, hidden_dims=[128, 64, 32]):
        super(E2EDelayModel, self).__init__()
        layers = []
        prev_dim = input_size

        # 动态构建隐藏层
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim

        self.feature_extractor = nn.Sequential(*layers)
        self.regressor = nn.Linear(prev_dim, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.regressor(features)


# 训练管理类
class NetworkTrainer:
    def __init__(self, model, dataset, batch_size=64):
        self.model = model
        self.dataloader = DataLoader(dataset, batch_size, shuffle=True)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for X_batch, y_batch in self.dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            self.scheduler.step(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")


# 使用示例
if __name__ == "__main__":
    # 数据准备
    dataset = NetworkDataset(num_samples=50000)

    # 模型初始化
    model = E2EDelayModel(hidden_dims=[256, 128, 64])

    # 训练配置
    trainer = NetworkTrainer(model, dataset, batch_size=128)
    trainer.train(epochs=100)

    # 测试推理
    test_sample = torch.FloatTensor([
        (50 - dataset.X_mean[0]) / dataset.X_std[0],  # 带宽50Mbps
        (20 - dataset.X_mean[1]) / dataset.X_std[1],  # 基础时延20ms
        (0.1 - dataset.X_mean[2]) / dataset.X_std[2]  # 丢包率10%
    ])

    with torch.no_grad():
        prediction = model(test_sample)
        denorm_pred = prediction.item() * dataset.y_std + dataset.y_mean
        print(f"Predicted E2E Delay: {denorm_pred:.2f}ms")
