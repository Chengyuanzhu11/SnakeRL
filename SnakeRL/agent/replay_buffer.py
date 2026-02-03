"""
经验回放缓冲区
"""
import random
from collections import deque
import numpy as np


class ReplayBuffer:
    """经验回放缓冲区，用于存储和采样训练经验"""
    
    def __init__(self, capacity=100000):
        """
        初始化缓冲区
        Args:
            capacity: 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        添加一条经验
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        随机采样一批经验
        Args:
            batch_size: 批次大小
        Returns:
            states, actions, rewards, next_states, dones
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """返回当前缓冲区大小"""
        return len(self.buffer)
