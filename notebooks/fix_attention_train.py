# Fix the train_step method in attention_dqn.py
with open('attention_dqn.py', 'r') as f:
    lines = f.readlines()

# Find and replace the problematic lines in train_step
new_lines = []
for i, line in enumerate(lines):
    # Replace the tensor conversion lines to match protagonist_dqn.py
    if 'states = torch.FloatTensor(np.array(states)).to(self.device)' in line:
        new_lines.append('        states = torch.FloatTensor(np.vstack(states)).to(self.device)\n')
    elif 'next_states = torch.FloatTensor(np.array(next_states)).to(self.device)' in line:
        new_lines.append('        next_states = torch.FloatTensor(np.vstack(next_states)).to(self.device)\n')
    else:
        new_lines.append(line)

with open('attention_dqn.py', 'w') as f:
    f.writelines(new_lines)

print("✓ Fixed attention_dqn.py train_step method")
