"""
Attention-Enhanced DQN for Stock Trading
Implements multi-head self-attention mechanism
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class AttentionBlock(nn.Module):
    """Multi-head self-attention block"""
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.head_dim = input_dim // num_heads
        
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        # Output projection
        self.out = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # x shape: (batch_size, input_dim)
        # Reshape for multi-head attention: (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        
        # Linear projections
        Q = self.query(x)  # (batch_size, 1, input_dim)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head: (batch_size, num_heads, 1, head_dim)
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention scores
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(x.device)
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        x = torch.matmul(attention, V)  # (batch_size, num_heads, 1, head_dim)
        
        # Concatenate heads
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, 1, self.input_dim)
        
        # Output projection
        x = self.out(x)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # (batch_size, input_dim)
        
        # Return both output and attention weights for visualization
        attention_weights = attention.mean(dim=1).squeeze(1).squeeze(1)  # Average over heads
        
        return x, attention_weights


class AttentionDQNNetwork(nn.Module):
    """DQN with attention mechanism"""
    def __init__(self, state_size, action_size, num_heads=4, hidden_size=128):
        super(AttentionDQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Feature embedding
        self.feature_embed = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Attention mechanism
        self.attention = AttentionBlock(hidden_size, num_heads=num_heads)
        
        # Post-attention processing
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output layer
        self.out = nn.Linear(hidden_size // 2, action_size)
        
    def forward(self, state):
        # Embed features
        x = self.feature_embed(state)
        
        # Apply attention
        x_att, attention_weights = self.attention(x)
        
        # Residual connection
        x = x + x_att
        
        # Further processing
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Q-values
        q_values = self.out(x)
        
        return q_values, attention_weights


class AttentionDQN:
    """Attention-Enhanced DQN Agent"""
    def __init__(self, state_size, action_size, attention_type='multi', num_heads=4,
                 learning_rate=0.001, gamma=0.95, epsilon_start=1.0, 
                 epsilon_end=0.01, epsilon_decay=0.995, memory_size=10000):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = AttentionDQNNetwork(
            state_size, action_size, num_heads=num_heads
        ).to(self.device)
        
        self.target_net = AttentionDQNNetwork(
            state_size, action_size, num_heads=num_heads
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Track attention weights for analysis
        self.attention_history = []
        
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values, attention_weights = self.policy_net(state_tensor)
            
            # Store attention weights for later analysis
            if training:
                self.attention_history.append(attention_weights.cpu().numpy())
                
                # Keep only recent history
                if len(self.attention_history) > 1000:
                    self.attention_history = self.attention_history[-1000:]
            
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        # Ensure all values are valid
        if state is None or next_state is None:
            return
        state = np.array(state) if not isinstance(state, np.ndarray) else state
        next_state = np.array(next_state) if not isinstance(next_state, np.ndarray) else next_state
        self.memory.append((state, action, reward, next_state, done))
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self, batch_size=32):
        """Train the network on a batch"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        
        # Validate batch (skip if any None values)
        batch = [(s, a, r, ns, d) for s, a, r, ns, d in batch 
                 if s is not None and ns is not None]
        if len(batch) < batch_size // 2:
            return None
        
        # Convert to tensors (same as protagonist_dqn)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
        
        # Reshape for network input
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        
        # Current Q values
        current_q_values, _ = self.policy_net(states)
        current_q_values = current_q_values.gather(1, actions)
        
        # Next Q values (using target network)
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_states)
            max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


if __name__ == "__main__":
    # Quick test
    print("Testing Attention DQN...")
    
    agent = AttentionDQN(
        state_size=13,
        action_size=3,
        num_heads=4,
        learning_rate=0.001
    )
    
    # Test forward pass
    test_state = np.random.randn(13)
    action = agent.select_action(test_state, training=False)
    print(f"✓ Selected action: {action}")
    
    # Test training
    for i in range(100):
        state = np.random.randn(13)
        next_state = np.random.randn(13)
        agent.store_experience(state, random.randint(0, 2), 
                              random.random(), next_state, False)
    
    loss = agent.train_step(batch_size=32)
    print(f"✓ Training step complete. Loss: {loss:.4f}")
    print(f"✓ Attention history size: {len(agent.attention_history)}")
    
    print("\n✓ Attention DQN ready!")
