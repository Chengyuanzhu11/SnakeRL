"""
贪吃蛇游戏环境 - Gym 风格实现
"""
import numpy as np
try:
    import pygame
except ImportError:
    import pygame_ce as pygame
from enum import Enum
from collections import namedtuple
import random
import os
import json

# 定义方向
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# 定义点
Point = namedtuple('Point', 'x, y')

# 颜色定义 - 高级配色方案
WHITE = (255, 255, 255)
BLACK = (15, 15, 25)
GRID_COLOR = (30, 30, 45)
BG_GRADIENT_TOP = (20, 20, 35)
BG_GRADIENT_BOTTOM = (10, 10, 20)

# 蛇的颜色 - 渐变青绿色
SNAKE_HEAD = (0, 230, 180)        # 青绿色蛇头
SNAKE_HEAD_GLOW = (100, 255, 220) # 蛇头高光
SNAKE_BODY_START = (0, 200, 150)  # 身体开始色
SNAKE_BODY_END = (0, 100, 80)     # 身体结束色
SNAKE_GLOW = (0, 255, 200)        # 发光效果
SNAKE_OUTLINE = (0, 80, 60)       # 轮廓色
SNAKE_EYE = (255, 255, 255)       # 眼睛白色
SNAKE_PUPIL = (20, 20, 20)        # 瞳孔

# 食物颜色 - 发光红色
FOOD_COLOR = (255, 60, 80)        # 鲜红色
FOOD_GLOW = (255, 100, 120)       # 外发光
FOOD_HIGHLIGHT = (255, 180, 180)  # 高光
FOOD_CORE = (255, 200, 200)       # 核心亮点

# 文字颜色
TEXT_COLOR = (220, 220, 230)
SCORE_COLOR = (255, 215, 0)       # 金色分数
HIGH_SCORE_COLOR = (180, 180, 255) # 最高分颜色

BLOCK_SIZE = 20


class SnakeEnv:
    """贪吃蛇强化学习环境"""
    
    def __init__(self, width=640, height=480, render_mode=False):
        self.width = width
        self.height = height
        self.render_mode = render_mode
        
        # 最高分文件路径
        self.high_score_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'high_score.json')
        self.high_score = self._load_high_score()
        
        # 初始化 pygame (如果需要渲染)
        if self.render_mode:
            pygame.init()
            self.font = pygame.font.SysFont('arial', 25)
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake RL')
            self.clock = pygame.time.Clock()
        
        self.reset()
    
    def reset(self):
        """重置环境"""
        # 初始化蛇的位置
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
        return self._get_state()
    
    def _place_food(self):
        """随机放置食物"""
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
    
    def step(self, action):
        """
        执行一步动作
        action: 0 = 直行, 1 = 右转, 2 = 左转
        返回: (state, reward, done, info)
        """
        self.frame_iteration += 1
        
        # 处理 pygame 事件 (如果渲染)
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
        # 保存之前的距离
        old_distance = self._get_distance_to_food()
        
        # 移动蛇
        self._move(action)
        self.snake.insert(0, self.head)
        
        # 检查游戏是否结束
        reward = 0
        done = False
        
        # 如果蛇撞墙或撞到自己，或者太久没吃到食物
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            done = True
            reward = -10
            return self._get_state(), reward, done, {'score': self.score}
        
        # 检查是否吃到食物
        if self.head == self.food:
            self.score += 1
            # 更新最高分并保存到文件
            if self.score > self.high_score:
                self.high_score = self.score
                self._save_high_score()
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            # 根据距离给予小奖励
            new_distance = self._get_distance_to_food()
            if new_distance < old_distance:
                reward = 1
            else:
                reward = -1
        
        # 渲染
        if self.render_mode:
            self._update_ui()
            self.clock.tick(15)
        
        return self._get_state(), reward, done, {'score': self.score}
    
    def _get_distance_to_food(self):
        """计算蛇头到食物的曼哈顿距离"""
        return abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
    
    def _is_collision(self, pt=None):
        """检查是否碰撞"""
        if pt is None:
            pt = self.head
        
        # 撞墙
        if pt.x > self.width - BLOCK_SIZE or pt.x < 0:
            return True
        if pt.y > self.height - BLOCK_SIZE or pt.y < 0:
            return True
        
        # 撞到自己
        if pt in self.snake[1:]:
            return True
        
        return False
    
    def _move(self, action):
        """
        根据动作移动蛇
        action: 0 = 直行, 1 = 右转, 2 = 左转
        """
        # 方向顺序: 右, 下, 左, 上
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if action == 0:  # 直行
            new_dir = clock_wise[idx]
        elif action == 1:  # 右转
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # 左转
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        
        self.head = Point(x, y)
    
    def _get_state(self):
        """
        获取当前状态 (11维向量)
        - 3个危险检测 (直行、右转、左转是否危险)
        - 4个方向 (当前朝向)
        - 4个食物位置 (上、下、左、右)
        """
        head = self.head
        
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        state = [
            # 直行危险
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)),
            
            # 右转危险
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d)),
            
            # 左转危险
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)),
            
            # 当前方向
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # 食物位置
            self.food.x < self.head.x,  # 食物在左边
            self.food.x > self.head.x,  # 食物在右边
            self.food.y < self.head.y,  # 食物在上边
            self.food.y > self.head.y   # 食物在下边
        ]
        
        return np.array(state, dtype=np.float32)
    
    def _update_ui(self):
        """更新游戏界面 - 高级流线型视觉"""
        # 渐变背景
        for y in range(self.height):
            ratio = y / self.height
            r = int(BG_GRADIENT_TOP[0] + (BG_GRADIENT_BOTTOM[0] - BG_GRADIENT_TOP[0]) * ratio)
            g = int(BG_GRADIENT_TOP[1] + (BG_GRADIENT_BOTTOM[1] - BG_GRADIENT_TOP[1]) * ratio)
            b = int(BG_GRADIENT_TOP[2] + (BG_GRADIENT_BOTTOM[2] - BG_GRADIENT_TOP[2]) * ratio)
            pygame.draw.line(self.display, (r, g, b), (0, y), (self.width, y))
        
        # 网格背景 (更精细)
        for x in range(0, self.width, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (0, y), (self.width, y), 1)
        
        # 获取蛇身体中心点列表
        centers = [(pt.x + BLOCK_SIZE // 2, pt.y + BLOCK_SIZE // 2) for pt in self.snake]
        
        # 第一层: 画蛇的发光效果 (外光晕)
        if len(centers) >= 2:
            pygame.draw.lines(self.display, (*SNAKE_GLOW[:3], 50), False, centers, 22)
        
        # 第二层: 画连接段之间的线条 (流线型身体)
        for i in range(len(centers) - 1):
            # 计算渐变色
            ratio = i / max(len(centers) - 1, 1)
            r = int(SNAKE_BODY_END[0] + (SNAKE_BODY_START[0] - SNAKE_BODY_END[0]) * (1 - ratio))
            g = int(SNAKE_BODY_END[1] + (SNAKE_BODY_START[1] - SNAKE_BODY_END[1]) * (1 - ratio))
            b = int(SNAKE_BODY_END[2] + (SNAKE_BODY_START[2] - SNAKE_BODY_END[2]) * (1 - ratio))
            body_color = (r, g, b)
            
            # 画粗线连接相邻节点
            pygame.draw.line(self.display, body_color, centers[i], centers[i + 1], 16)
        
        # 第三层: 画每个节点的圆形 (平滑过渡)
        for i, center in enumerate(centers):
            ratio = i / max(len(centers) - 1, 1)
            r = int(SNAKE_BODY_END[0] + (SNAKE_BODY_START[0] - SNAKE_BODY_END[0]) * (1 - ratio))
            g = int(SNAKE_BODY_END[1] + (SNAKE_BODY_START[1] - SNAKE_BODY_END[1]) * (1 - ratio))
            b = int(SNAKE_BODY_END[2] + (SNAKE_BODY_START[2] - SNAKE_BODY_END[2]) * (1 - ratio))
            body_color = (r, g, b)
            
            # 主体圆
            radius = 9 if i > 0 else 10
            pygame.draw.circle(self.display, body_color, center, radius)
            
            # 高光效果
            highlight_offset = (-2, -2)
            highlight_pos = (center[0] + highlight_offset[0], center[1] + highlight_offset[1])
            highlight_color = (min(r + 60, 255), min(g + 60, 255), min(b + 60, 255))
            pygame.draw.circle(self.display, highlight_color, highlight_pos, 3)
        
        # 蛇头特殊处理
        head = self.snake[0]
        head_center = (head.x + BLOCK_SIZE // 2, head.y + BLOCK_SIZE // 2)
        
        # 蛇头发光
        pygame.draw.circle(self.display, SNAKE_GLOW, head_center, 14)
        pygame.draw.circle(self.display, SNAKE_HEAD, head_center, 11)
        pygame.draw.circle(self.display, SNAKE_HEAD_GLOW, (head_center[0] - 3, head_center[1] - 3), 4)
        
        # 蛇眼睛 (根据方向)
        if self.direction == Direction.RIGHT:
            eye1 = (head.x + BLOCK_SIZE - 5, head.y + 6)
            eye2 = (head.x + BLOCK_SIZE - 5, head.y + BLOCK_SIZE - 8)
        elif self.direction == Direction.LEFT:
            eye1 = (head.x + 4, head.y + 6)
            eye2 = (head.x + 4, head.y + BLOCK_SIZE - 8)
        elif self.direction == Direction.UP:
            eye1 = (head.x + 6, head.y + 4)
            eye2 = (head.x + BLOCK_SIZE - 8, head.y + 4)
        else:  # DOWN
            eye1 = (head.x + 6, head.y + BLOCK_SIZE - 6)
            eye2 = (head.x + BLOCK_SIZE - 8, head.y + BLOCK_SIZE - 6)
        
        # 眼白
        pygame.draw.circle(self.display, SNAKE_EYE, eye1, 4)
        pygame.draw.circle(self.display, SNAKE_EYE, eye2, 4)
        # 瞳孔
        pygame.draw.circle(self.display, SNAKE_PUPIL, eye1, 2)
        pygame.draw.circle(self.display, SNAKE_PUPIL, eye2, 2)
        
        # 食物 (发光球体效果)
        food_center = (self.food.x + BLOCK_SIZE // 2, self.food.y + BLOCK_SIZE // 2)
        
        # 外发光
        pygame.draw.circle(self.display, FOOD_GLOW, food_center, 14)
        # 主体
        pygame.draw.circle(self.display, FOOD_COLOR, food_center, 10)
        # 内层高光
        pygame.draw.circle(self.display, FOOD_HIGHLIGHT, (food_center[0] - 2, food_center[1] - 2), 5)
        # 核心亮点
        pygame.draw.circle(self.display, FOOD_CORE, (food_center[0] - 3, food_center[1] - 3), 2)
        
        # 显示分数 (现代风格)
        score_text = f"SCORE  {self.score}"
        shadow = self.font.render(score_text, True, (0, 0, 0))
        text = self.font.render(score_text, True, SCORE_COLOR)
        self.display.blit(shadow, [12, 12])
        self.display.blit(text, [10, 10])
        
        # 显示最高分
        high_score_text = f"BEST  {self.high_score}"
        shadow2 = self.font.render(high_score_text, True, (0, 0, 0))
        text2 = self.font.render(high_score_text, True, HIGH_SCORE_COLOR)
        self.display.blit(shadow2, [12, 42])
        self.display.blit(text2, [10, 40])
        
        pygame.display.flip()
    
    def render(self):
        """渲染当前帧"""
        if not self.render_mode:
            return
        self._update_ui()
    
    def close(self):
        """关闭环境"""
        if self.render_mode:
            pygame.quit()
    
    def _load_high_score(self):
        """从文件加载最高分"""
        try:
            if os.path.exists(self.high_score_file):
                with open(self.high_score_file, 'r') as f:
                    data = json.load(f)
                    return data.get('high_score', 0)
        except:
            pass
        return 0
    
    def _save_high_score(self):
        """保存最高分到文件"""
        try:
            with open(self.high_score_file, 'w') as f:
                json.dump({'high_score': self.high_score}, f)
        except:
            pass
