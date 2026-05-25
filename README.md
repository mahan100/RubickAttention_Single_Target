# RubickAttention

**A Feature‑Specific Doubly Stochastic Attention Mechanism for Long‑Term Time Series Forecasting**

This repository contains the official implementation of the paper  
*"RubickAttention: A Feature‑Specific Doubly Stochastic Attention Mechanism for Long‑Term Time Series Forecasting"*.

## Overview

RubickAttention is a novel attention block designed for **univariate long‑term time series forecasting (LTSF)**.  
Instead of feeding a flat sequence to a Transformer, RubickAttention **reshapes the lookback window into a 2D matrix** of *artificial features* (e.g., 12 time steps → 6×2).  
It then learns **per‑feature gating weights** that scale each artificial feature independently, using a **learnable context weight matrix**.  
An **alternating softmax normalisation** pushes the attention matrix towards **doubly stochastic** properties, stabilising training and improving interpretability.

## Key Features

- **Feature‑specific gating** – each artificial feature receives its own gating weight, computed from all features but applied element‑wise.  
- **Multi‑head architecture** with learnable head‑scalar selectors – the model can up‑ or down‑weight entire heads based on context.  
- **Alternating softmax (Rubick Axis)** – cyclic row‑wise / column‑wise normalisation encourages doubly stochastic attention, preventing rank collapse and improving interpretability.  
- **Strong empirical performance** – state‑of‑the‑art results on **ETTh1, ETTh2, ILI, and Weather** benchmarks, with particularly large gains on long horizons.

## Repository Structure

All code (model definition, training, evaluation) is contained in self‑contained Jupyter notebooks – one per dataset:

Each notebook includes:
- Data loading and preprocessing (following the PatchTST benchmark protocol)
- The full RubickAttention model implementation (layer classes + training loop)
- Training and validation loss / MAE logging
- Test evaluation and result tables

## Model Architecture
Lookback window (q) → reshape(S × F) → Learnable Tanh transformation → Multi‑Head Attention (context‑weighted gating) → Concatenation → Alternating softmax (Rubick Axis) → Forecasting Head (Dense + Conv1D) → Prediction

## Citation
@article{mohseni2025rubickattention,
  - title   = {RubickAttention: A Feature‑Specific Doubly Stochastic Attention Mechanism for Long‑Term Time Series Forecasting},
  - author  = {Mehdi Mohseni Mahani},
  - journal = {soon....},
  - year    = {2025}
}

## Datasets

The model is evaluated on five widely used LTSF benchmarks:

- **ETTh1, ETTh2** (Electricity Transformer Temperature)  
- **Exchange** (daily currency rates)  
- **ILI** (Influenza‑Like Illness)  
- **Weather** (21 meteorological variables)  

Data loaders follow the PatchTST benchmark protocol for fair comparison.

## Results Highlights

| Dataset | Horizon | MSE | MAE | Improvement over PatchTST |
|---------|---------|-----|-----|---------------------------|
| ETTh1   | 96      | 0.0664 | 0.1965 | 82% MSE reduction |
| ETTh2   | 96      | 0.1309 | 0.2801 | 55% MSE reduction |
| Weather | 96      | 0.0050 | 0.0577 | 95% MSE reduction |
| ILI     | 24 (weeks) | 1.1716 | 0.9401 | 42% MSE reduction |

Full per‑horizon results are available in the paper and inside each notebook.

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/RubickAttention.git
cd RubickAttention
```

## License
This version clearly states that there’s no `requirements.txt`, and dependencies are handled inline within the notebooks. It keeps all the paper’s highlights and is ready for your GitHub repo.
