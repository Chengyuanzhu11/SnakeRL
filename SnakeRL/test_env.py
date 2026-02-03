"""测试脚本"""
import sys
print("Python version:", sys.version)

try:
    from environment.snake_env import SnakeEnv
    print("SnakeEnv imported successfully!")
    
    env = SnakeEnv()
    state = env.reset()
    print("State shape:", state.shape)
    print("State:", state)
    
    # 测试几步
    for i in range(5):
        action = 0  # 直行
        next_state, reward, done, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward}, done={done}, score={info['score']}")
        if done:
            break
    
    print("\nEnvironment test PASSED!")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\nError: {e}")
