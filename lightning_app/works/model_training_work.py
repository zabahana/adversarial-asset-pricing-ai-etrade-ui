from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from . import LightningWork, HAS_LIGHTNING


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism for sequence modeling."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Store residual
        residual = query
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(attention_output)
        output = self.layer_norm(output + residual)
        
        return output, attention_weights


class MHADQN(nn.Module):
    """Multi-Head Attention Deep Q-Network for stock price prediction."""
    
    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 20,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        hidden_sizes: List[int] = [256, 128],
        dropout_rate: float = 0.1,
        output_size: int = 3,  # BUY, HOLD, SELL
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Final projection layers
        final_layers = []
        prev_size = d_model
        
        for hidden_size in hidden_sizes:
            final_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),  # Use LayerNorm instead of BatchNorm to handle batch_size=1
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        final_layers.append(nn.Linear(prev_size, output_size))
        self.final_layers = nn.Sequential(*final_layers)
        
    def forward(self, x, return_attention=False):
        """Forward pass through MHA-DQN.
        
        Args:
            x: Input tensor (batch_size, seq_len, features)
            return_attention: If True, return attention weights for explainability
            
        Returns:
            q_values: Q-values for actions (batch_size, output_size)
            attention_weights: Dict of attention weights per layer (if return_attention=True)
        """
        batch_size, seq_len, features = x.shape
        
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Store attention weights for explainability
        attention_weights = {}
        
        # Multi-head attention layers
        for i, (attention_layer, layer_norm) in enumerate(zip(self.attention_layers, self.layer_norms)):
            # Self-attention
            attn_output, attn_weights = attention_layer(x, x, x)
            
            # Store attention weights (average across batch for visualization)
            if return_attention:
                # attn_weights shape: (batch_size, num_heads, seq_len, seq_len)
                # Average across batch and store
                attention_weights[f'layer_{i+1}'] = attn_weights.cpu().numpy()
            
            # Residual connection and layer norm
            x = layer_norm(x + attn_output)
            
            # Feed-forward network
            ff_output = self.feed_forward(x)
            x = layer_norm(x + ff_output)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, d_model)
        
        # Final layers
        q_values = self.final_layers(x)  # (batch_size, output_size)
        
        if return_attention:
            return q_values, attention_weights
        return q_values


class ModelTrainingWork(LightningWork):
    """Trains MHA-DQN models (clean and adversarial) from features."""
    
    def __init__(self, model_dir: str, cache_dir: str) -> None:
        if HAS_LIGHTNING:
            super().__init__(parallel=True)
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        (self.model_dir / "mha_dqn").mkdir(parents=True, exist_ok=True)
        
    def run(
        self,
        ticker: str,
        feature_path: str,
        train_clean: bool = True,
        train_adversarial: bool = True,
        num_episodes: int = 50,
        batch_size: int = 32,
        sequence_length: int = 20,
        risk_level: str = "Medium",
    ) -> Dict[str, str]:
        """Train MHA-DQN models from features."""
        
        print(f"[1] Starting model training for {ticker}")
        print(f"[CONFIG] Risk Level: {risk_level}")
        print(f"[2] Features: {feature_path}")
        print(f"[3] Episodes: {num_episodes}")
        print(f"[4] Batch size: {batch_size}")
        print(f"[5] Sequence length: {sequence_length}")
        
        # Load features
        features = pd.read_parquet(feature_path)
        print(f"[6] Loaded {len(features)} rows of features")
        
        if len(features) < sequence_length + 10:
            raise ValueError(f"Not enough data: need at least {sequence_length + 10} rows, got {len(features)}")
        
        # Create sequences and targets
        sequences, targets = self._create_sequences(features, sequence_length)
        print(f"[7] Created {len(sequences)} sequences for training")
        
        # Get input dimension
        input_dim = sequences.shape[-1]
        print(f"[8] Input dimension: {input_dim}")
        
        results = {}
        
        # Train clean model
        if train_clean:
            print(f"\n[9] Training CLEAN MHA-DQN model...")
            clean_model = self._train_model(
                sequences, targets, input_dim, sequence_length,
                num_episodes=num_episodes, batch_size=batch_size,
                adversarial=False
            )
            clean_path = self.model_dir / "mha_dqn" / "clean.ckpt"
            torch.save(clean_model.state_dict(), clean_path)
            print(f"[10] Clean model saved to: {clean_path}")
            results["clean_model"] = str(clean_path)
        
        # Train adversarial model
        if train_adversarial:
            print(f"\n[11] Training ADVERSARIAL ROBUST MHA-DQN model...")
            adv_model = self._train_model(
                sequences, targets, input_dim, sequence_length,
                num_episodes=num_episodes, batch_size=batch_size,
                adversarial=True
            )
            adv_path = self.model_dir / "mha_dqn" / "adversarial.ckpt"
            torch.save(adv_model.state_dict(), adv_path)
            print(f"[12] Adversarial model saved to: {adv_path}")
            results["adversarial_model"] = str(adv_path)
        
        print(f"\n[COMPLETE] Training complete!")
        return results
    
    def _create_sequences(
        self, features: pd.DataFrame, sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences and targets from features."""
        
        # Select numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        if "return" not in numeric_cols:
            # Calculate return if not present
            if "close" in features.columns:
                features["return"] = features["close"].pct_change()
            else:
                # Use first numeric column as proxy
                features["return"] = features[numeric_cols[0]].pct_change()
            numeric_cols.append("return")
        
        # Normalize features
        feature_data = features[numeric_cols].values
        feature_mean = np.nanmean(feature_data, axis=0, keepdims=True)
        feature_std = np.nanstd(feature_data, axis=0, keepdims=True) + 1e-8
        feature_data = (feature_data - feature_mean) / feature_std
        
        # Replace NaN with 0
        feature_data = np.nan_to_num(feature_data, nan=0.0)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(feature_data)):
            seq = feature_data[i - sequence_length:i]
            # Target: next day return (for action prediction)
            next_return = feature_data[i, numeric_cols.index("return")] if "return" in numeric_cols else 0.0
            
            # Convert return to action: BUY (2) if return > 0.01, SELL (0) if return < -0.01, else HOLD (1)
            if next_return > 0.01:
                target = 2  # BUY
            elif next_return < -0.01:
                target = 0  # SELL
            else:
                target = 1  # HOLD
            
            sequences.append(seq)
            targets.append(target)
        
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.int64)
        
        return sequences, targets
    
    def _train_model(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        input_dim: int,
        sequence_length: int,
        num_episodes: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        adversarial: bool = False,
        adv_epsilon: float = 0.01,
    ) -> MHADQN:
        """Train MHA-DQN model."""
        
        # Initialize model
        model = MHADQN(
            input_dim=input_dim,
            sequence_length=sequence_length,
            d_model=128,
            num_heads=8,
            num_layers=3,
            hidden_sizes=[256, 128],
            dropout_rate=0.1,
            output_size=3,  # BUY, HOLD, SELL
        )
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Convert to tensors
        sequences_tensor = torch.FloatTensor(sequences)
        targets_tensor = torch.LongTensor(targets)
        
        # Training loop
        model.train()
        for episode in range(num_episodes):
            total_loss = 0.0
            num_batches = 0
            
            # Shuffle data
            indices = np.random.permutation(len(sequences))
            
            for i in range(0, len(sequences), batch_size):
                batch_indices = indices[i:i + batch_size]
                # Skip batches with only 1 sample to ensure stable training (LayerNorm works with size 1, but batches of 2+ are preferred)
                if len(batch_indices) < 2:
                    continue
                batch_sequences = sequences_tensor[batch_indices]
                batch_targets = targets_tensor[batch_indices]
                
                # Forward pass
                q_values = model(batch_sequences)
                
                # Calculate loss
                loss = criterion(q_values, batch_targets)
                
                # Adversarial training: add small perturbation
                if adversarial:
                    batch_sequences.requires_grad_(True)
                    q_values_adv = model(batch_sequences)
                    loss_adv = criterion(q_values_adv, batch_targets)
                    
                    # Compute gradient
                    model.zero_grad()
                    loss_adv.backward()
                    
                    # Add adversarial perturbation
                    with torch.no_grad():
                        perturbation = adv_epsilon * torch.sign(batch_sequences.grad)
                        batch_sequences_adv = batch_sequences + perturbation
                        batch_sequences_adv = torch.clamp(batch_sequences_adv, -1.0, 1.0)
                    
                    # Forward pass with adversarial example
                    q_values_adv = model(batch_sequences_adv)
                    loss_adv = criterion(q_values_adv, batch_targets)
                    
                    # Combined loss
                    loss = (loss + loss_adv) / 2
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            if (episode + 1) % 10 == 0:
                print(f"[INFO] Episode {episode + 1}/{num_episodes}, Loss: {avg_loss:.4f}")
        
        model.eval()
        return model

