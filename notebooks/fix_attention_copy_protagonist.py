# Copy the exact tensor conversion approach from protagonist_dqn.py
with open('attention_dqn.py', 'r') as f:
    lines = f.readlines()

# Find the train_step method and replace tensor conversion
new_lines = []
in_train_step = False
converted = False

for i, line in enumerate(lines):
    if 'def train_step(self, batch_size=32):' in line:
        in_train_step = True
    
    # Replace the problematic lines
    if in_train_step and 'Convert to tensors' in line and not converted:
        # Replace the next 5 lines with protagonist_dqn.py approach
        new_lines.append(line)  # Keep the comment
        new_lines.append('        states = torch.FloatTensor([e[0] for e in batch])\n')
        new_lines.append('        actions = torch.LongTensor([e[1] for e in batch])\n')
        new_lines.append('        rewards = torch.FloatTensor([e[2] for e in batch])\n')
        new_lines.append('        next_states = torch.FloatTensor([e[3] for e in batch])\n')
        new_lines.append('        dones = torch.FloatTensor([e[4] for e in batch])\n')
        new_lines.append('        \n')
        new_lines.append('        # Move to device\n')
        new_lines.append('        states = states.to(self.device)\n')
        new_lines.append('        actions = actions.to(self.device)\n')
        new_lines.append('        rewards = rewards.to(self.device)\n')
        new_lines.append('        next_states = next_states.to(self.device)\n')
        new_lines.append('        dones = dones.to(self.device)\n')
        
        # Skip the old lines
        j = i + 1
        while j < len(lines) and 'Current Q values' not in lines[j]:
            j += 1
        converted = True
        continue
    
    if converted and i < j:
        continue
        
    new_lines.append(line)

with open('attention_dqn.py', 'w') as f:
    f.writelines(new_lines)

print("✓ Fixed attention_dqn.py to use protagonist_dqn.py tensor conversion")
