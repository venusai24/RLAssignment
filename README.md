# Adversarial RL for Alert Prioritization in Fraud Detection

This project implements a game-theoretic framework for robust alert prioritization in fraud detection, combining **Adversarial Reinforcement Learning (ARL)** with the **Double Oracle** algorithm.

## Project Overview

The core objective is to compute an optimal stochastic policy for a defender to prioritize alerts, assuming a strong adversary who knows the defender's policy and dynamically chooses optimal attacks. The interaction is modeled as a zero-sum game, solved iteratively to find an approximate Mixed-Strategy Nash Equilibrium (MSNE).

## Dataset

The implementation uses the **Credit Card Fraud Detection** dataset from Kaggle.
- **Dataset URL**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Description**: Contains transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred in two days, where 482 frauds are recorded out of 284,807 transactions.

> [!IMPORTANT]
> Due to file size limits, the `creditcard.csv` file is not included in the repository. Please download it from the link above and place it in the root directory before running the processing scripts.

## Project Structure

```text
.
├── AlertDetectionEnv.py         # Shared game-theoretic environment logic
├── DoubleOracleSolver.py        # Shared Double Oracle algorithm implementation
├── paper_implementation/        # Original paper implementation
│   ├── DDPG.py                  # DDPG-MIX Oracle (Actor-Critic)
│   └── verify_fraud_metrics.py  # Script to verify paper's fraud case study results
├── improvement_experiments/     # Extended experiments and improvements
│   ├── SGS.py                   # Semi-Gradient n-step Sarsa Oracle
│   └── verify_fraud_metrics_sgs_real.py # Comparison between DDPG and SGS
└── process_fraud_data.py        # Data preprocessing and parameter derivation
```

## Setup and Installation

1. **Install Dependencies**:
   ```bash
   pip install numpy pandas tensorflow scikit-learn scipy
   ```

2. **Prepare Data**:
   Download `creditcard.csv` from Kaggle and place it in the root directory.

## How to Run

### 1. Verify Original Paper Results
To run the DDPG-based prioritization as described in the paper:
```bash
python3 paper_implementation/verify_fraud_metrics.py
```

### 2. Run Comparison Experiments
To compare the paper's DDPG method against the SGS (Semi-Gradient n-step Sarsa) improvement:
```bash
python3 improvement_experiments/verify_fraud_metrics_sgs_real.py
```

## Key Hyperparameters & Information Alignment

This implementation is strictly aligned with the paper's specifications:
- **Information Constraints**: The defender operates under **Partial Observability**, seeing only alert counts ($N$), while the adversary has full visibility ($N, M, S$).
- **Network Architecture**: 
    - **Actor**: 16 hidden units (Fraud), Tanh activation, Xavier initialization.
    - **Critic**: 32 hidden units (Fraud), ReLU hidden activation, Linear output, He Normal initialization.
- **RL Parameters**: Discount factor 0.95, Actor LR 0.001, Critic LR 0.002, Buffer 40,000.

## Acknowledgments
This implementation is based on the research paper: 
**Finding Needles in a Moving Haystack: Prioritizing Alerts with Adversarial Reinforcement Learning**
