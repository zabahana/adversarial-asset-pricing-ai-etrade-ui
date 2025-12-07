# Explanation: 3 Attention Layers Architecture

## Overview
The "3 ATTENTION LAYERS (Stacked Multi-Head Attention)" section represents three identical layers stacked on top of each other. Each layer processes the data sequentially, with each layer building on the representations learned by the previous layer.

## Structure of Each Attention Layer

Each of the 3 layers follows this structure (from bottom to top):

```
┌─────────────────────────────────────┐
│  1. Multi-Head Attention            │  ← Analyzes relationships between time steps
└──────────────┬──────────────────────┘
               │ (Residual connection)
               ↓
┌─────────────────────────────────────┐
│  2. Layer Normalization             │  ← Normalizes the output for stability
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  3. Feed-Forward Network            │  ← Non-linear transformation
│     (128 → 512 → 128 dimensions)    │
└──────────────┬──────────────────────┘
               │ (Residual connection)
               ↓
┌─────────────────────────────────────┐
│  4. Layer Normalization             │  ← Final normalization
└─────────────────────────────────────┘
```

## Detailed Breakdown

### 1. Multi-Head Attention
- **Purpose**: Analyzes relationships and dependencies between different time steps in the sequence
- **Input**: Sequence of feature vectors (128 dimensions each)
- **Process**: 
  - Each of the 8 attention heads focuses on different aspects of the relationships
  - Computes attention weights to determine which time steps are most relevant
  - Creates a weighted combination of all time steps
- **Output**: Enhanced representation where each time step now contains information from relevant other time steps

### 2. Layer Normalization (First)
- **Purpose**: Stabilizes training by normalizing the output from attention
- **Process**: Normalizes across the feature dimensions (128 dimensions)
- **Why needed**: Prevents activations from growing too large, which can cause training instability

### 3. Feed-Forward Network (128 → 512 → 128)
- **Purpose**: Applies non-linear transformations to extract complex patterns
- **Structure**:
  ```
  Input:  128 dimensions
     ↓
  Linear Layer: 128 → 512 dimensions (expansion)
     ↓
  ReLU Activation: Adds non-linearity
     ↓
  Dropout: Prevents overfitting
     ↓
  Linear Layer: 512 → 128 dimensions (compression back)
     ↓
  Output: 128 dimensions
  ```
- **Why 512?**: Expanding to 512 dimensions allows the network to learn more complex patterns. The expansion factor of 4 (128 × 4 = 512) is a common pattern in transformer architectures.
- **Why compress back?**: Returns to 128 dimensions to maintain consistent dimensionality for stacking multiple layers

### 4. Layer Normalization (Second)
- **Purpose**: Normalizes the output from the feed-forward network
- **Ensures**: Stable gradients for the next layer (or final output)

## Why 3 Layers?

1. **Layer 1**: Learns basic temporal relationships and short-term patterns
2. **Layer 2**: Builds on Layer 1's representations to capture medium-term dependencies
3. **Layer 3**: Integrates information from Layers 1 and 2 to understand long-term patterns and complex relationships

## Residual Connections

Each layer uses **residual connections** (skip connections) that allow:
- Original information to flow through unchanged
- The layer's transformations to be additive improvements
- Easier training of deep networks (gradients can flow through residuals)

**Pattern**: `output = LayerNorm(input + transformed_input)`

## Complete Flow Through 3 Layers

```
Input from Attention Heads (8 heads output)
    ↓
[Layer 1] Multi-Head Attention → Layer Norm → Feed-Forward (128→512→128) → Layer Norm
    ↓
[Layer 2] Multi-Head Attention → Layer Norm → Feed-Forward (128→512→128) → Layer Norm
    ↓
[Layer 3] Multi-Head Attention → Layer Norm → Feed-Forward (128→512→128) → Layer Norm
    ↓
Output to Global Average Pooling
```

## Key Points

- **Stacked**: Each layer processes the output of the previous layer, allowing for hierarchical feature learning
- **Identical Structure**: All 3 layers have the same internal structure but learn different patterns
- **128 → 512 → 128**: The feed-forward network expands, transforms, then compresses
- **Layer Norm**: Applied twice per layer for training stability
- **Residual Connections**: Enable deep network training by preserving gradient flow

This architecture allows the model to capture both short-term market patterns and long-term trends in the financial data.

