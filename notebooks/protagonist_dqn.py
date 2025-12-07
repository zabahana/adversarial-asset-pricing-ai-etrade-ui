import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for trading decisions
    """
    def __init__(self, state_size, action_size, hidden_sizes=[128, 128, 64]):
        super(DQNNetwork, self).__init__()
        
        # Build network layers dynamically
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        
        # Output layer - Q-values for each action
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, state):
        """Forward pass"""
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for training stability
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store experience"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch"""
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        
        # Handle None next_states (when episode ends)
        next_states_list = []
        for e in batch:
            if e[3] is None:
                # Use zeros for terminal states
                next_states_list.append(np.zeros_like(e[0]))
            else:
                next_states_list.append(e[3])
        next_states = torch.FloatTensor(np.array(next_states_list))
        
        dones = torch.BoolTensor([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class ProtagonistDQN:
    """
    Protagonist DQN Trading Agent
    
    Actions:
        0: HOLD - Keep current position
        1: BUY  - Buy shares (if have cash)
        2: SELL - Sell shares (if have shares)
    
    State:
        - Price features (close, sma_3, sma_6, sma_12)
        - Momentum features (momentum_3m, momentum_6m, momentum_12m)
        - Risk metrics (volatility, sharpe_12m, max_drawdown_12m)
        - Position info (shares_held, cash_balance)
    """
    
    def __init__(self, state_size, action_size=3, learning_rate=0.001,
                 gamma=0.95, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, buffer_size=10000):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.loss_history = []
        self.q_value_history = []
        
        print("=" * 80)
        print("PROTAGONIST DQN AGENT INITIALIZED")
        print("=" * 80)
        print(f"State size: {state_size}")
        print(f"Action size: {action_size}")
        print(f"Device: {self.device}")
        print(f"Network architecture:")
        print(self.q_network)
        print("=" * 80)
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: If False, always exploit (no exploration)
        
        Returns:
            action: 0 (HOLD), 1 (BUY), or 2 (SELL)
        """
        # Exploration vs Exploitation
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randrange(self.action_size)
        
        # Exploit: best action based on Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            self.q_value_history.append(q_values.max().item())
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self, batch_size=32):
        """
        Perform one training step using experience replay
        """
        # Need enough experiences
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q-values (using target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss (Huber loss for stability)
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Store loss
        self.loss_history.append(loss.item())
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network (soft update)"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint"""
        checkpoint = {
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'loss_history': self.loss_history,
            'q_value_history': self.q_value_history
        }
        torch.save(checkpoint, filepath)
        print(f"✓ Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        self.loss_history = checkpoint['loss_history']
        self.q_value_history = checkpoint['q_value_history']
        print(f"✓ Checkpoint loaded from {filepath}")
    
    def get_action_name(self, action):
        """Get human-readable action name"""
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        return action_names.get(action, "UNKNOWN")


class TradingEnvironment:
    """
    Trading environment for the DQN agent
    """
    
    def __init__(self, data, initial_capital=100000, transaction_cost=0.001,
                 shares_per_trade=100):
        """
        Args:
            data: DataFrame with columns [timestamp, symbol, close_price, features...]
            initial_capital: Starting cash
            transaction_cost: Transaction cost as percentage (0.001 = 0.1%)
            shares_per_trade: Number of shares to buy/sell per action
        """
        self.data = data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.shares_per_trade = shares_per_trade
        
        # Feature columns (exclude timestamp, symbol) - MUST BE SET BEFORE reset()
        self.feature_cols = [col for col in data.columns 
                           if col not in ['timestamp', 'symbol', 'ingestion_timestamp']]
        
        # State - NOW we can reset
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares_held = 0
        self.portfolio_value = self.initial_capital
        self.total_trades = 0
        
        # Performance tracking
        self.portfolio_history = [self.initial_capital]
        self.trade_history = []
        
        return self._get_state()
    
    def _get_state(self):
        """
        Get current state representation
        
        State includes:
        - Market features (price, momentum, volatility, etc.)
        - Position information (normalized)
        """
        if self.current_step >= len(self.data):
            return None
        
        # Get current row
        row = self.data.iloc[self.current_step]
        
        # Market features
        market_features = []
        for col in self.feature_cols:
            value = row[col]
            # Handle NaN values
            if pd.isna(value):
                market_features.append(0.0)
            else:
                market_features.append(float(value))
        
        # Position features (normalized)
        current_price = row['close_price']
        position_features = [
            self.shares_held / 1000,  # Normalize shares
            self.cash / self.initial_capital,  # Normalize cash
            self.portfolio_value / self.initial_capital,  # Normalize portfolio value
        ]
        
        state = np.array(market_features + position_features, dtype=np.float32)
        
        return state
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done, info)
        
        Actions:
            0: HOLD
            1: BUY
            2: SELL
        """
        if self.current_step >= len(self.data):
            return None, 0, True, {}
        
        # Get current price
        current_price = self.data.iloc[self.current_step]['close_price']
        
        # Execute action
        reward = 0
        trade_executed = False
        
        if action == 1:  # BUY
            cost = self.shares_per_trade * current_price * (1 + self.transaction_cost)
            if cost <= self.cash:
                self.cash -= cost
                self.shares_held += self.shares_per_trade
                self.total_trades += 1
                trade_executed = True
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'BUY',
                    'price': current_price,
                    'shares': self.shares_per_trade
                })
        
        elif action == 2:  # SELL
            if self.shares_held >= self.shares_per_trade:
                proceeds = self.shares_per_trade * current_price * (1 - self.transaction_cost)
                self.cash += proceeds
                self.shares_held -= self.shares_per_trade
                self.total_trades += 1
                trade_executed = True
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': self.shares_per_trade
                })
        
        # Move to next step
        self.current_step += 1
        
        # Calculate new portfolio value
        if self.current_step < len(self.data):
            next_price = self.data.iloc[self.current_step]['close_price']
            new_portfolio_value = self.cash + self.shares_held * next_price
        else:
            new_portfolio_value = self.cash + self.shares_held * current_price
        
        # Calculate reward (change in portfolio value)
        reward = (new_portfolio_value - self.portfolio_value) / self.initial_capital
        
        # Penalty for excessive trading
        if trade_executed:
            reward -= 0.0001  # Small penalty
        
        # Update portfolio value
        self.portfolio_value = new_portfolio_value
        self.portfolio_history.append(self.portfolio_value)
        
        # Check if done
        done = self.current_step >= len(self.data)
        
        # Get next state
        next_state = self._get_state() if not done else None
        
        # Info
        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'shares_held': self.shares_held,
            'total_trades': self.total_trades
        }
        
        return next_state, reward, done, info
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        
        metrics = {
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'final_value': self.portfolio_value,
            'total_trades': self.total_trades,
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(12),
            'max_drawdown': self._calculate_max_drawdown(self.portfolio_history)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, values):
        """Calculate maximum drawdown"""
        cumulative = np.array(values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


if __name__ == "__main__":
    import pandas as pd
    
    print("=" * 80)
    print("PROTAGONIST DQN TRADING AGENT - TEST")
    print("=" * 80)
    
    # Example state size (adjust based on your features)
    # Assuming: 10 market features + 3 position features
    state_size = 13
    
    # Create agent
    agent = ProtagonistDQN(
        state_size=state_size,
        action_size=3,
        learning_rate=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    # Test action selection
    print("\n📊 Testing action selection...")
    dummy_state = np.random.randn(state_size)
    action = agent.select_action(dummy_state)
    print(f"Selected action: {action} ({agent.get_action_name(action)})")
    
    # Test training step
    print("\n🎯 Testing training step...")
    for _ in range(100):
        state = np.random.randn(state_size)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.randn(state_size)
        done = False
        agent.store_experience(state, action, reward, next_state, done)
    
    loss = agent.train_step(batch_size=32)
    if loss is not None:
        print(f"Training loss: {loss:.4f}")
    
    print("\n✓ Protagonist DQN agent ready for training!")
    print("=" * 80)

