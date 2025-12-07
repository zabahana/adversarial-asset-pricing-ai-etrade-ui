#!/usr/bin/env python3
"""Script to add section numbering, dividers, and algorithm section to streamlit_app.py"""

import re

def add_algorithm_section():
    """Algorithm section content for MHA-DQN framework"""
    return '''    # Section 5: Algorithm & Framework
    st.markdown('<div class="section-divider-thick"></div>', unsafe_allow_html=True)
    st.markdown("""    
    <div class="section-card">
        <h3 style="color: #1e40af; font-family: 'Inter', sans-serif; font-size: 1.25rem; font-weight: 700; margin: 0 0 1.5rem 0; display: flex; align-items: center;">
            <span class="section-number">5</span>
            Algorithm & Framework
        </h3>
    </div>
    <div style="padding: 0 2rem;">
    """, unsafe_allow_html=True)
    
    # Display Architecture Diagram
    st.markdown('<h4 style="color: #3b82f6; font-family: \\'Inter\\', sans-serif; font-size: 1rem; font-weight: 600; margin: 1.5rem 0 0.75rem 0;">Multi-Head Attention Architecture</h4>', unsafe_allow_html=True)
    try:
        architecture_html = _generate_mha_architecture_diagram()
        if architecture_html:
            st.markdown(architecture_html, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not display architecture diagram: {str(e)}")
    
    # Algorithm Pseudocode
    st.markdown('<h4 style="color: #3b82f6; font-family: \\'Inter\\', sans-serif; font-size: 1rem; font-weight: 600; margin: 2rem 0 0.75rem 0;">MHA-DQN Algorithm</h4>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; font-family: 'Inter', monospace; font-size: 0.875rem; line-height: 1.8; color: #111827;">
        <pre style="margin: 0; white-space: pre-wrap; font-family: 'Inter', monospace;">
<strong>Algorithm: MHA-DQN Adversarial-Robust Trading Agent</strong>

<strong>Input:</strong> Historical price data, features X, action space A = {BUY, HOLD, SELL}
<strong>Output:</strong> Q-values Q(s, a), trading policy π

<strong>1. Feature Engineering</strong>
   - Extract 8 feature groups (Price, Macro, Commodities, Indices, Forex, Technical, Earnings, Crypto)
   - Normalize features: X_norm = (X - μ) / σ
   - Create sequence windows: S_t = [X_{t-L+1}, ..., X_t] where L = 20

<strong>2. Multi-Head Attention Encoding</strong>
   For each feature group g in {1, ..., 8}:
     - Project features: H_g = X_g · W_proj (128 dims)
     - Apply 8 attention heads in parallel:
       Q_g = H_g · W_q, K_g = H_g · W_k, V_g = H_g · W_v
       Attention_g = softmax(Q_g · K_g^T / √d_k) · V_g
     - Concatenate heads: MHA_g = Concat([Attention_{g,1}, ..., Attention_{g,8}])
   
   Stack 3 attention layers with residual connections:
     H_{l+1} = LayerNorm(MHA_l(H_l) + H_l)
     H_{l+1} = LayerNorm(FFN(H_{l+1}) + H_{l+1})
   where FFN: Linear(128 → 512) → ReLU → Linear(512 → 128)

<strong>3. Global Average Pooling</strong>
   - Pool across sequence: z = GlobalAvgPool(H_3)
   - Output dimension: 128

<strong>4. Q-Value Computation</strong>
   - Final layers: z → Linear(128 → 64) → ReLU → Linear(64 → 3)
   - Q-values: Q(s, a) for a in {SELL, HOLD, BUY}

<strong>5. Adversarial Training (FGSM)</strong>
   For each training batch:
     - Clean forward pass: Q_clean = MHA-DQN(X)
     - Compute gradient: ∇_X L(Q_clean, y)
     - Generate adversarial: X_adv = X + ε · sign(∇_X L)
     - Adversarial forward pass: Q_adv = MHA-DQN(X_adv)
     - Combined loss: L = 0.5 · L(Q_clean, y) + 0.5 · L(Q_adv, y)
     - Update: θ ← θ - α · ∇_θ L

<strong>6. Trading Policy</strong>
   - Action selection: a* = argmax_a Q(s, a)
   - Position sizing: size = risk_level · portfolio_value · Q(s, a*) / Σ Q(s, a)
   - Execute: BUY/SELL/HOLD based on a*

<strong>7. Experience Replay & Target Network</strong>
   - Store transitions: (s, a, r, s', done) in replay buffer D
   - Sample batch: B ~ Uniform(D)
   - Target Q: Q_target = r + γ · max_a' Q_target(s', a')
   - Update Q-network to minimize TD-error: (Q(s, a) - Q_target)²

<strong>Key Parameters:</strong>
   - Attention heads: 8
   - Attention layers: 3
   - Sequence length: 20 days
   - Feature dimensions: 128
   - FGSM epsilon (ε): 0.01
   - Learning rate (α): 0.001
   - Discount factor (γ): 0.99
        </pre>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
'''

if __name__ == "__main__":
    with open('streamlit_app.py', 'r') as f:
        content = f.read()
    
    # Find where to insert algorithm section (after forecast section closes, before performance)
    # Look for "Section 4: Performance Metrics"
    section4_pos = content.find('# Section 4: Performance Metrics and Backtesting')
    
    if section4_pos > 0:
        # Insert algorithm section before Section 4
        algorithm_code = add_algorithm_section()
        content = content[:section4_pos] + algorithm_code + '\n' + content[section4_pos:]
        
        # Update Section 4 number to Section 6
        content = content.replace('# Section 4: Performance Metrics and Backtesting', 
                                 '# Section 6: Performance Metrics and Backtesting')
        content = re.sub(r'<span class="section-number">4</span>\s*Performance Metrics',
                        '<span class="section-number">6</span>Performance Metrics', content)
    
    with open('streamlit_app.py', 'w') as f:
        f.write(content)
    
    print("✅ Added algorithm section and updated section numbers")

