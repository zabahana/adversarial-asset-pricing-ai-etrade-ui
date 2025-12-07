import re

with open('attention_dqn.py', 'r') as f:
    content = f.read()

# Find and replace the entire train_step method
old_pattern = r'    def train_step\(self, batch_size=32\):.*?return loss\.item\(\)'
new_train_step = '''    def train_step(self, batch_size=32):
        """Train the network on a batch"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors - ensure consistent shapes
        states = torch.FloatTensor(np.array([np.array(s).flatten() for s in states])).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([np.array(s).flatten() for s in next_states])).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
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
        
        return loss.item()'''

content = re.sub(old_pattern, new_train_step, content, flags=re.DOTALL)

with open('attention_dqn.py', 'w') as f:
    f.write(content)

print("✓ Fixed train_step method in attention_dqn.py")
