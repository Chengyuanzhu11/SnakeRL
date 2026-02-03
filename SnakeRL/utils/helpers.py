"""
辅助函数
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curve(scores, mean_scores, save_path=None):
    """
    绘制训练曲线
    Args:
        scores: 每回合得分列表
        mean_scores: 平均得分列表
        save_path: 保存路径 (可选)
    """
    plt.figure(figsize=(12, 5))
    
    # 得分曲线
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.6, label='Score')
    plt.plot(mean_scores, label='Mean Score', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 滑动窗口平均
    plt.subplot(1, 2, 2)
    if len(scores) >= 100:
        window_avg = np.convolve(scores, np.ones(100)/100, mode='valid')
        plt.plot(window_avg, label='100-episode moving average')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
        plt.title('Moving Average (window=100)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training curve saved to {save_path}")
    
    plt.show()


def moving_average(data, window=100):
    """
    计算滑动平均
    Args:
        data: 数据列表
        window: 窗口大小
    Returns:
        滑动平均值列表
    """
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')
