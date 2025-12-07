with open('protagonist_dqn.py', 'r') as f:
    content = f.read()

# Find and replace the sample method
old_sample = '''    def sample(self, batch_size):
        """Sample random batch"""
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones'''

new_sample = '''    def sample(self, batch_size):
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
        
        return states, actions, rewards, next_states, dones'''

content = content.replace(old_sample, new_sample)

with open('protagonist_dqn.py', 'w') as f:
    f.write(content)

print("✓ Fixed replay buffer to handle terminal states")
