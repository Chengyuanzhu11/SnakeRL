"""
贪吃蛇 DQN 演示脚本
"""
import argparse
import os
import sys
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.snake_env import SnakeEnv
from agent.dqn_agent import DQNAgent


def play(args):
    """使用训练好的模型玩贪吃蛇"""
    print("=" * 50)
    print("Snake RL - Demo Mode")
    print("=" * 50)
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("Please train a model first using: python train.py")
        return
    
    # 创建环境和智能体
    env = SnakeEnv(render_mode=True)
    agent = DQNAgent(
        state_size=11,
        action_size=3,
        epsilon_start=0.0,  # 不探索
        epsilon_end=0.0
    )
    
    # 加载模型
    agent.load(args.model)
    agent.epsilon = 0.0  # 确保不探索
    print(f"Loaded model from: {args.model}")
    print("=" * 50)
    
    total_score = 0
    
    for game in range(1, args.games + 1):
        state = env.reset()
        done = False
        
        while not done:
            # 选择动作 (贪婪策略)
            action = agent.select_action(state, training=False)
            
            # 执行动作
            state, reward, done, info = env.step(action)
            
            # 控制游戏速度
            time.sleep(0.05)
        
        score = info['score']
        total_score += score
        print(f"Game {game}: Score = {score}")
    
    print("=" * 50)
    print(f"Average Score: {total_score / args.games:.2f}")
    print("=" * 50)
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Snake RL Demo')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                        help='Path to model file (default: checkpoints/best_model.pth)')
    parser.add_argument('--games', type=int, default=1,
                        help='Number of games to play (default: 1)')
    
    args = parser.parse_args()
    play(args)


if __name__ == '__main__':
    main()
