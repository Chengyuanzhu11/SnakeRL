"""
贪吃蛇 DQN 训练脚本
"""
import argparse
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.snake_env import SnakeEnv
from agent.dqn_agent import DQNAgent
from utils.helpers import plot_training_curve


def train(args):
    """训练 DQN 智能体"""
    print("=" * 50)
    print("Snake RL - DQN Training")
    print("=" * 50)
    print(f"Episodes: {args.episodes}")
    print(f"Render: {args.render}")
    print("=" * 50)
    
    # 创建环境和智能体
    env = SnakeEnv(render_mode=args.render)
    agent = DQNAgent(
        state_size=11,
        action_size=3,
        hidden_size=256,
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        target_update=100
    )
    
    # 训练记录
    scores = []
    mean_scores = []
    total_score = 0
    best_score = 0
    
    # 创建模型保存目录
    os.makedirs('checkpoints', exist_ok=True)
    
    for episode in range(1, args.episodes + 1):
        state = env.reset()
        score = 0
        done = False
        
        while not done:
            # 选择动作
            action = agent.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.store_experience(state, action, reward, next_state, done)
            
            # 训练
            agent.train_step()
            
            state = next_state
            score = info['score']
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 更新记录
        scores.append(score)
        total_score += score
        mean_score = total_score / episode
        mean_scores.append(mean_score)
        
        # 保存最佳模型
        if score > best_score:
            best_score = score
            agent.save('checkpoints/best_model.pth')
        
        # 打印进度
        if episode % 10 == 0:
            print(f"Episode {episode:4d} | Score: {score:3d} | "
                  f"Mean: {mean_score:.2f} | Best: {best_score:3d} | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        # 定期保存模型
        if episode % 100 == 0:
            agent.save(f'checkpoints/model_ep{episode}.pth')
    
    # 保存最终模型
    agent.save('checkpoints/final_model.pth')
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best Score: {best_score}")
    print(f"Final Mean Score: {mean_scores[-1]:.2f}")
    print("=" * 50)
    
    # 绘制训练曲线
    plot_training_curve(scores, mean_scores, save_path='training_curve.png')
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Snake RL Training')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes (default: 500)')
    parser.add_argument('--render', action='store_true',
                        help='Render the game during training')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
