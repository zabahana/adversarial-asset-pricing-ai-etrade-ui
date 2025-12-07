# Model Planning: Adversarial-Robust Asset Pricing AI System

## 1. Selected Machine Learning Models

### Primary Model: Deep Reinforcement Learning (DQN with Adversarial Training)

**Model Type:** Deep Q-Network (DQN) with Adversarial Robustness
**Model Name:** `AdversarialDQN_AssetPricing_v1`

**Architecture Components:**
- **Input Layer:** State representation (20 features: technical indicators + market data)
- **Hidden Layers:** 
  - Dense Layer 1: 128 neurons, ReLU activation
  - Dropout: 0.3 (for regularization)
  - Dense Layer 2: 64 neurons, ReLU activation
  - Dropout: 0.2
  - Dense Layer 3: 32 neurons, ReLU activation
- **Output Layer:** Q-values for 3 actions (Buy/Hold/Sell)

**Cost Function:** Huber Loss (smooth L1 loss)
- More robust to outliers than MSE
- Formula: L(y, f(x)) = 0.5(y - f(x))² for |y - f(x)| ≤ δ, else δ(|y - f(x)| - 0.5δ)

**Optimization:** Adam Optimizer
- Learning Rate: 0.001 (with decay)
- β1 = 0.9, β2 = 0.999

**Activation Functions:**
- Hidden Layers: ReLU (prevents vanishing gradients)
- Output Layer: Linear (Q-value estimation)

**Weight Initialization:** He initialization
- Suitable for ReLU activation
- Prevents vanishing/exploding gradients

### Secondary Model: Adversarial Attack Generator

**Model Name:** `FGSM_AttackGenerator_v1`
**Purpose:** Generate adversarial examples for robustness training

**Attack Method:** Fast Gradient Sign Method (FGSM)
- Perturbation: ε * sign(∇_x J(θ, x, y))
- Epsilon range: [0.01, 0.1]

### Baseline Models for Comparison

**Model Name:** `LinearRegression_Baseline`
- Simple linear regression on price prediction
- Used to establish minimum performance threshold

**Model Name:** `RandomForest_Baseline`
- 100 estimators, max_depth=10
- Traditional ML approach for comparison

## 2. Model Selection Justification

### Why Deep Reinforcement Learning?

**Sequential Decision Making:**
- Asset pricing requires sequential decisions over time
- RL naturally handles temporal dependencies
- Learns optimal pricing strategy through interaction

**Adaptability:**
- Markets are non-stationary; RL adapts to changing conditions
- Continuous learning from new data
- Handles complex state-action spaces

**Risk-Reward Optimization:**
- RL balances immediate and long-term rewards
- Natural fit for financial decision-making
- Incorporates transaction costs and market impact

### Why Adversarial Training?

**Robustness to Market Manipulation:**
- Financial markets subject to spoofing, wash trading
- Adversarial training improves resilience
- Prevents exploitation by malicious actors

**Real-World Applicability:**
- Production systems must handle noisy, corrupted data
- Adversarial examples simulate worst-case scenarios
- Improves generalization to unseen market conditions

## 3. Model Assumptions and Constraints

### Assumptions

**Market Efficiency (Weak Form):**
- Past prices contain information for future predictions
- Technical analysis has predictive power
- Not assuming strong-form efficiency

**Data Quality:**
- Market data is accurate and timely
- Missing data can be reasonably imputed
- Outliers represent real market events, not errors

**Stationarity (Local):**
- Market dynamics remain relatively stable within training windows
- Major regime changes are detectable
- Model retraining handles structural breaks

### Constraints

**Computational Resources:**
- Training time: <6 hours per model iteration
- Inference latency: <100ms for real-time pricing
- Memory: <4GB GPU RAM

**Data Availability:**
- Minimum 30 days historical data for meaningful training
- At least 5 features (technical indicators)
- Daily or intraday granularity

**Regulatory Requirements:**
- Model decisions must be explainable
- No use of insider information
- Compliant with financial regulations

**Adversarial Robustness:**
- Model must maintain >80% accuracy under ε=0.05 perturbations
- Should not catastrophically fail under worst-case attacks
- Graceful degradation under increasing attack strength

## 4. Model Hyperparameters

### DQN Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Replay Buffer Size | 10,000 | Balance memory usage and experience diversity |
| Batch Size | 32 | Computational efficiency and stable learning |
| Learning Rate | 0.001 | Standard starting point with adaptive decay |
| Discount Factor (γ) | 0.95 | Balance immediate and future rewards |
| Epsilon (exploration) | 1.0 → 0.01 | Start with exploration, decay to exploitation |
| Epsilon Decay | 0.995 | Gradual transition over ~1000 episodes |
| Target Network Update | Every 100 steps | Stabilize training |
| Gradient Clipping | [-1, 1] | Prevent exploding gradients |

### Adversarial Training Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Attack Epsilon | 0.01-0.1 | Realistic perturbation range |
| Attack Ratio | 0.5 | 50% adversarial, 50% clean examples |
| Attack Type | FGSM + PGD | Fast and strong attacks |
| Robustness Threshold | 0.80 | Acceptable accuracy under attack |

## 5. Feature Engineering Plan

### Input Features (20 dimensions)

**Price-Based Features (5):**
1. Close Price (normalized)
2. 5-day Price Change %
3. 10-day Price Change %
4. High-Low Range %
5. Volume Ratio (current/average)

**Technical Indicators (10):**
6. SMA_5 (5-day simple moving average)
7. SMA_10 (10-day simple moving average)
8. SMA_20 (20-day simple moving average)
9. EMA_12 (12-day exponential moving average)
10. RSI_14 (14-day relative strength index)
11. MACD Line
12. MACD Signal
13. Bollinger Band Upper
14. Bollinger Band Lower
15. Volatility (20-day rolling std)

**Market Context (5):**
16. Trading Volume (normalized)
17. Time of Day (encoded)
18. Day of Week (encoded)
19. Market Trend (bull/bear/neutral)
20. VIX Index (market volatility proxy)

### Feature Scaling Strategy

**Normalization Method:** Min-Max Scaling [0, 1]
- Suitable for neural networks
- Preserves zero entries
- Formula: x_scaled = (x - x_min) / (x_max - x_min)

**Standardization:** Z-score for specific features
- Applied to volume, volatility
- Formula: z = (x - μ) / σ

## 6. Model Evaluation Metrics

### Primary Metrics

**Cumulative Return:**
- Total profit/loss over evaluation period
- Benchmark: Buy-and-hold strategy

**Sharpe Ratio:**
- Risk-adjusted returns
- Formula: (R_p - R_f) / σ_p
- Target: >1.0

**Maximum Drawdown:**
- Largest peak-to-trough decline
- Risk measure
- Target: <20%

### Adversarial Robustness Metrics

**Robust Accuracy:**
- Accuracy under adversarial attacks
- Target: >80% at ε=0.05

**Attack Success Rate:**
- Percentage of attacks that fool the model
- Target: <30%

**Certified Robustness:**
- Provable bounds on model robustness
- Using randomized smoothing

## 7. Training Strategy

### Phase 1: Baseline Training (2 weeks)
- Train on clean historical data
- Establish performance baseline
- Tune hyperparameters

### Phase 2: Adversarial Training (2 weeks)
- Introduce adversarial examples
- Gradually increase attack strength
- Monitor robustness metrics

### Phase 3: Fine-tuning (1 week)
- Optimize for production deployment
- Reduce model size if needed
- Final validation on holdout set

### Phase 4: Production Testing (1 week)
- Real-time inference testing
- Stress testing under load
- A/B testing against baseline
