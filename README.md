# Anomaly Detection in Stock Price Time-Series Using an LSTM Autoencoder

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Model](https://img.shields.io/badge/Model-LSTM%20Autoencoder-purple)

This project implements an **unsupervised LSTM Autoencoder** for detecting anomalies in financial time-series data. It focuses on learning the normal temporal behavior of **GOOG (Google)** closing stock prices and identifying unusual market movements using reconstruction error.

This work was completed as part of my **Industrial Training & Defence Internship** at the **Indian Institute of Technology Bhubaneswar**.

---

## Features
- Unsupervised anomaly detection (no labels required)
- LSTM Autoencoder for temporal pattern learning
- Reconstruction error–based anomaly scoring
- Sliding window–based time-series preprocessing
- Clean and modular implementation
- Threshold-based anomaly classification

---

## Problem Statement

Financial stock prices change dynamically due to market sentiment, global events, and investor behavior. Detecting unusual or suspicious movements is important for risk monitoring and market analysis.

An **LSTM Autoencoder** is used here to:
1. Learn normal stock price sequences
2. Reconstruct them
3. Detect anomalies when reconstruction fails significantly

---

## Methodology

### **1. Dataset**
- Source: Yahoo Finance  
- Data Used: **Close price**
- Preprocessing:
  - Datetime indexing
  - Standardization using StandardScaler
  - Sliding window segmentation (30 time steps)
  - 80:20 train–test split

---

### **2. Model Architecture**
The autoencoder consists of:

#### **Encoder**
- LSTM (128 units)
- Dropout (0.2)

#### **Latent Space**
- Dense compressed representation of sequence

#### **Decoder**
- RepeatVector
- LSTM (128 units)
- TimeDistributed(Dense(1))

#### **Training Setup**
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam (lr = 0.001)
- Batch Size: 32
- Epochs: 100 (Early Stopping triggered at 12)

---

## Anomaly Detection Strategy

### **Reconstruction Error**
The model reconstructs input sequences.  
Abnormal sequences (unseen patterns) produce **higher reconstruction error**.

$$
E = \frac{1}{T} \sum_{t=1}^{T} |x_t - \hat{x_t}|
$$

### **Threshold Selection**
A statistical threshold is computed using train reconstruction errors:

$$
\theta = \mu_E + 2\sigma_E
$$

If  

$$
E > \theta \quad \text{→ Anomaly}
$$


---

## Results (Summary)

- Model achieved **low reconstruction loss** on normal patterns.
- On test data, reconstruction errors spiked during:
  - Sudden trend reversals  
  - High volatility regions  
  - Sharp upward/downward jumps  
- Threshold-based evaluation detected multiple anomaly points corresponding to market fluctuations.

This confirms that the LSTM Autoencoder effectively distinguishes normal vs abnormal stock movements based on sequence reconstruction behavior.

---

