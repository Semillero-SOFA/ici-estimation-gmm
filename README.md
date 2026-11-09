# Interchannel Interference Estimation in Nyquist-WDM with GMMs and Deep Learning

A comprehensive research framework for estimating **Interchannel Interference (ICI)** and spectral channel spacing in Nyquist-WDM optical networks using **Gaussian Mixture Models (GMM)** feature extraction coupled with Deep Neural Networks (DNN).

## 🔬 Methodology Overview

1. **Constellation Modeling:** Employs Gaussian Mixture Models (GMM) to characterize the statistical distortion and centroid displacement of received QPSK/QAM constellation clusters under varying channel spacing.
2. **Feature Engineering:** Extracts mixture weights, covariance matrices, and cluster centroids as low-dimensional representation vectors describing ICI severity.
3. **Deep Regression:** Trains deep feedforward neural networks (DNN) using extracted GMM parameters to accurately regress optical carrier frequency spacing and predict channel degradation across **16 GBaud** and **32 GBaud** transmission setups.

## 📂 Repository Structure

```
├── 16GBd/                    # Notebooks and experiments for 16 GBaud transmission data
├── 32GBd/                    # Notebooks and experiments for 32 GBaud transmission data
├── gmm_utils.py              # GMM fitting, parameter extraction, and feature transformations
├── sofa.py                   # Shared SOFA DSP and optical signal processing utilities
└── README.md                 # Project documentation
```

## 🛠️ Requirements & Tech Stack

- Python 3.9+
- Scikit-learn (GaussianMixture, metrics)
- PyTorch / TensorFlow for DNN regressors
- NumPy, SciPy, Matplotlib

## 👥 Authors & Affiliation

- **Author:** César Alejandro Montoya Gómez
- **Affiliation:** Semillero de Óptica y Fotónica Aplicada (SOFA) / Grupo GITA, Universidad de Antioquia (UdeA).
