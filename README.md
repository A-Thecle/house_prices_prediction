# 🏡 House Price Prediction with Neural Networks

This project predicts house prices using a deep learning model built with **PyTorch** and tracked with **MLflow**.

---

## 📊 Overview

The model predicts house prices based on real estate features.

- 📦 Dataset size: **21,613 samples**
- 🧩 Features: **21**
- 📐 Shape: `(21613, 21)`

### 📏 Evaluation Metrics
- Loss
- Mean Squared Error (MSE)
- R² Score

---

## ⚙️ Project Workflow

### 🔍 1. EDA & Preprocessing
- Data exploration
- Handling missing values
- Feature scaling
- Train/test split

### 🧠 2. Model Architecture
Input → 16 neurons (ReLU + Dropout 0.3)
→ 8 neurons (ReLU + Dropout 0.3)
→ 1 output neuron


### 🚀 3. Training
- Optimizer: **Adam**
- Iterations: **5,000**
- Loss Function: **MSE**
- Tracking: **MLflow (metrics & artifacts)**

---

## 📈 Results

- 📉 Loss and MSE decrease over iterations
- 🎯 R² ≈ **0.8** on test set
- ✅ Good predictive performance achieved

---

## 📊 Visualization

Training results are visualized using:
- Loss curves
- MSE curves
- R² score evolution

📁 Saved as: `house_price.png`

---

## 🔗 MLflow Integration

MLflow is used to track:
- 📦 Artifacts
- 📊 Metrics
- 🤖 Trained model

---

## 🛠️ Requirements

```bash
Python 3.x
PyTorch
MLflow
matplotlib
scikit-learn
tqdm
