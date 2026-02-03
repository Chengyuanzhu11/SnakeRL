"""
DQN 智能体
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.q_network import QNetwork
from agent.replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN 智能体"""
    
    def __init__(
        self,
        state_size=11,
        action_size=3,
        hidden_size=256,
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update=10,
        device=None
    ):
        """
        初始化 DQN 智能体
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        # 设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Q 网络和目标网络
        self.q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放
        self.memory = ReplayBuffer(buffer_size)
    
    def select_action(self, state, training=True):
        """
        选择动作 (ε-greedy)
        Args:
            state: 当前状态
            training: 是否训练模式
        Returns:
            动作索引
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state)
        self.q_network.train()
        
        return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """执行一步训练"""
        if len(self.memory) < self.batch_size:
            return None
        
        # 采样经验
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前 Q 值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标 Q 值
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失
        loss = nn.MSELoss()(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
