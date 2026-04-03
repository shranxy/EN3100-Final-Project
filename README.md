# Using Machine Learning Algorithms for Predicting Remaining Useful Life (RUL) of an Induction Motor

**EN3100 Final Year Project — Cardiff University, School of Engineering**

**Author:** Maximillian A. Hall Barr (23037697)
**Supervisor:** Dr. Michael Packianather
**Degree:** BEng Mechanical Engineering
**Date:** April 2026

## Overview

This project compares four machine learning and deep learning models for predicting the Remaining Useful Life (RUL) of induction motor bearings:

- **Linear Regression** — baseline
- **XGBoost** — gradient boosted ensemble
- **1D-CNN** — convolutional neural network on raw vibration/sensor data
- **CNN-LSTM** — hybrid convolutional-recurrent architecture

Models were evaluated on two datasets:
- **MFPT Bearing Fault Dataset** (primary) — 21 vibration recordings, 11,133 windows, 12 engineered features
- **NASA N-CMAPSS DS02** (validation) — 6.5 million turbofan engine samples, 18 sensor features

## Key Results

| Model | MFPT RMSE | N-CMAPSS RMSE |
|-------|-----------|---------------|
| Linear Regression | 78.58 | 11.81 |
| XGBoost | 30.39 | 9.26 |
| **1D-CNN** | **11.60** | **5.85** |
| CNN-LSTM | 13.90 | 5.81 |

- 1D-CNN achieved 62% improvement over XGBoost on bearing data
- CNN-LSTM underperformed CNN on bearing data (19.8% worse) and tied on turbofan data
- SHAP analysis identified kurtosis (bearings) and T24 temperature (turbofan) as dominant features
- Deep learning consistently outperformed traditional ML across both datasets

## Repository Structure
