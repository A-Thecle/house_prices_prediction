#House Price Prediction with Neural Networks
This project predicts house prices using a neural network built with PyTorch and tracked with MLflow.

#Overview
The model predicts house prices based on input features. The dataset contains 21 features and 21,613 samples (shape = (21613, 21)). We evaluate the model using loss, Mean Squared Error (MSE), and R² score.

#Steps
-EDA & Preprocessing
  Explore data, handle missing values, scale features, and split into training/testing sets.
-Model Architecture
  Input → 16 neurons (ReLU + Dropout 0.3) → 8 neurons (ReLU + Dropout 0.3) → 1 output neuron
-Training
  Optimizer: Adam
  Iterations: 5,000
  Loss: MSE
  Metrics logged with MLflow


#Results
Loss and MSE decrease over iterations
R² ≈ 0.8 on test set
Good predictive performance achieved

#Visualization
Plots for training/test loss, R², and MSE are saved as house_price.png

#MLflow Integration
artifact,  metrics, and model are logged for experiment tracking

Requirements
Python 3.x, PyTorch, MLflow, matplotlib, scikit-learn, tqdm
