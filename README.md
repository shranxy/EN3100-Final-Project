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

Notebooks - Kaggle notebooks for both experiments
MFPT Figures - figures from MFPT experiment
N-CMAPSS Figures - figures from N-CMAPSS experiment
Report - Final dissertation

## Notebooks

Both notebooks run on Kaggle with GPU acceleration (Tesla T4).

- `mfpt-ml-dl.ipynb` — Full MFPT bearing pipeline: data loading, synthetic RUL construction, feature extraction, all four models, SHAP analysis
- `n-cmapss-ml-dl.ipynb` — Full N-CMAPSS turbofan pipeline: HDF5 loading, engine-level splitting, windowing, all four models, SHAP analysis, error analysis

### To reproduce:
1. Create a new Kaggle notebook with GPU enabled
2. Add the relevant dataset (MFPT or N-CMAPSS DS02) to your notebook inputs
3. Upload the notebook and run all cells

## Datasets

Datasets are not included in this repository due to size. They are publicly available:

- **MFPT:** [mfpt.org/fault-data-sets](https://www.mfpt.org/fault-data-sets/) or search "MFPT Fault Data Sets" on Kaggle
- **N-CMAPSS:** [NASA Prognostics Data Repository](https://data.nasa.gov/dataset/N-CMAPSS-DS02) or search "N-CMAPSS" on Kaggle

## Technologies

- Python 3.12
- TensorFlow / Keras
- XGBoost (CUDA accelerated)
- scikit-learn
- SHAP
- NumPy, pandas, matplotlib, seaborn

## License

This project was completed as part of the EN3100 module at Cardiff University School of Engineering. Code is provided for academic reference.
