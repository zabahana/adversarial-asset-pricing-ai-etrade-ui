import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        
        # Neural network for Q-values
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class AdversarialTraining:
    def __init__(self, agent, attack_strength=0.1):
        self.agent = agent
        self.attack_strength = attack_strength
    
    def generate_adversarial_state(self, state):
        """Generate adversarial perturbations to the state"""
        noise = np.random.normal(0, self.attack_strength, state.shape)
        adversarial_state = state + noise
        return np.clip(adversarial_state, -1, 1)  # Clip to reasonable bounds
    
    def adversarial_training_step(self, state, action, reward, next_state, done):
        """Train with both original and adversarial examples"""
        # Train on original data
        self.agent.remember(state, action, reward, next_state, done)
        
        # Train on adversarial data
        adv_state = self.generate_adversarial_state(state)
        adv_next_state = self.generate_adversarial_state(next_state)
        self.agent.remember(adv_state, action, reward, adv_next_state, done)
