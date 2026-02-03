# Snake RL - 贪吃蛇强化学习项目

使用 DQN (Deep Q-Network) 算法训练 AI 玩贪吃蛇游戏。

## 项目结构

```
SnakeRL/
├── environment/         # 游戏环境
│   └── snake_env.py
├── agent/               # DQN 智能体
│   ├── dqn_agent.py
│   └── replay_buffer.py
├── models/              # 神经网络模型
│   └── q_network.py
├── utils/               # 辅助工具
│   └── helpers.py
├── train.py             # 训练脚本
├── play.py              # 演示脚本
└── requirements.txt     # 依赖包
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练

```bash
python train.py
```

训练参数:
- `--episodes`: 训练回合数 (默认: 500)
- `--render`: 是否显示训练过程

### 演示

```bash
python play.py
```

加载训练好的模型并可视化游戏过程。

## 算法说明

- **状态表示**: 11维向量 (危险检测、当前方向、食物相对位置)
- **动作空间**: 3个动作 (直行、左转、右转)
- **奖励设计**: 吃到食物 +10, 死亡 -10, 靠近食物 +1, 远离食物 -1
- **网络结构**: 3层全连接网络 (256-256-3)

## 技术栈

- PyTorch - 深度学习框架
- Pygame - 游戏可视化
- NumPy - 数值计算
