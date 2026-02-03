"""
Q 网络模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """深度 Q 网络"""
    
    def __init__(self, state_size=11, action_size=3, hidden_size=256):
        """
        初始化 Q 网络
        Args:
            state_size: 状态维度
            action_size: 动作数量
            hidden_size: 隐藏层大小
        """
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入状态
        Returns:
            Q 值
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
