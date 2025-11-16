# Anomaly-Prediction-using-LSTM-Autoencoders-in-Stock-price
An unsupervised LSTM Autoencoder model for detecting anomalies in financial time-series. Uses GOOG stock data to learn normal price patterns and identify unusual market movements via reconstruction error. Includes preprocessing, model training, thresholding, and visualization.

# Features

Unsupervised anomaly detection

LSTM Autoencoder for sequence reconstruction

Statistical thresholding for anomaly labeling

Visual anomaly marking on time-series plots

Modular and clean implementation

Model Architecture
Encoder

LSTM (128 units)

Dropout (0.2)

Latent Vector

Dense compressed representation

Decoder

RepeatVector

LSTM (128 units)

TimeDistributed(Dense(1))

Training Details

Loss: MSE

Optimizer: Adam (lr = 0.001)

Batch Size: 32

Epochs: 100 (Early Stopping at 12)
