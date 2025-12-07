import torch
import numpy as np
import pandas as pd
from protagonist_dqn import ProtagonistDQN, TradingEnvironment

print("=" * 80)
print("TESTING PROTAGONIST DQN AGENT")
print("=" * 80)

# 1. Test agent creation
print("\n1. Creating DQN Agent...")
state_size = 13  # 10 market features + 3 position features
agent = ProtagonistDQN(state_size=state_size, action_size=3)

# 2. Test action selection
print("\n2. Testing Action Selection...")
dummy_state = np.random.randn(state_size)
for i in range(5):
    action = agent.select_action(dummy_state, training=True)
    print(f"   Step {i+1}: {agent.get_action_name(action)} (epsilon={agent.epsilon:.3f})")

# 3. Test experience replay
print("\n3. Testing Experience Replay...")
for _ in range(100):
    state = np.random.randn(state_size)
    action = np.random.randint(0, 3)
    reward = np.random.randn()
    next_state = np.random.randn(state_size)
    done = False
    agent.store_experience(state, action, reward, next_state, done)

print(f"   Buffer size: {len(agent.replay_buffer)}")

# 4. Test training
print("\n4. Testing Training Step...")
loss = agent.train_step(batch_size=32)
print(f"   Training loss: {loss:.4f}")

# 5. Test epsilon decay
print("\n5. Testing Epsilon Decay...")
for i in range(10):
    agent.decay_epsilon()
print(f"   Epsilon after 10 decays: {agent.epsilon:.4f}")

# 6. Test with dummy trading environment
print("\n6. Testing Trading Environment...")

# Create dummy market data
dates = pd.date_range('2023-01-01', periods=100, freq='ME')
dummy_data = pd.DataFrame({
    'timestamp': dates,
    'symbol': 'TEST',
    'close_price': 100 + np.cumsum(np.random.randn(100) * 2),
    'monthly_return': np.random.randn(100) * 0.02,
    'volatility': np.abs(np.random.randn(100) * 0.01),
    'sharpe_12m': np.random.randn(100) * 0.5 + 1.0,
    'momentum_3m': np.random.randn(100) * 0.05,
    'momentum_6m': np.random.randn(100) * 0.08,
    'momentum_12m': np.random.randn(100) * 0.12,
    'max_drawdown_12m': -np.abs(np.random.randn(100) * 0.1),
    'vol_ratio': np.random.rand(100) + 0.5,
    'return_to_vol': np.random.randn(100) * 0.5
})

env = TradingEnvironment(dummy_data, initial_capital=100000)

# Run a few steps
state = env.reset()
print(f"   Initial portfolio: ${env.portfolio_value:,.2f}")

for i in range(10):
    action = agent.select_action(state, training=False)  # Use greedy policy
    next_state, reward, done, info = env.step(action)
    
    if done:
        break
    
    state = next_state
    
    if i % 3 == 0:
        print(f"   Step {i+1}: {agent.get_action_name(action)} | "
              f"Portfolio: ${info['portfolio_value']:,.2f} | "
              f"Reward: {reward:.4f}")

print(f"\n   Final portfolio: ${env.portfolio_value:,.2f}")
metrics = env.calculate_metrics()
print(f"   Total return: {metrics['total_return']*100:.2f}%")
print(f"   Total trades: {metrics['total_trades']}")

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED!")
print("=" * 80)
print("\nProtagonist DQN Agent is ready!")
print("\nNext steps:")
print("  1. Collect monthly data for 10 stocks")
print("  2. Train the protagonist on real market data")
print("  3. Add adversarial training later")
print("=" * 80)
