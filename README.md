# GoFundMe Fraud Detection System

## Overview

Machine learning system to detect fraudulent fundraisers using metadata and text features.

## Installation

1. Clone repository
2. Create virtual environment: `python -m venv venv`
3. Activate environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost  textstat textblob html5lib`

## Usage

### Option 1: Run Notebook

1. Start Jupyter: `jupyter notebook`
2. Open `notebooks/gfm_fraud_detection.ipynb`
3. Run all cells

### Option 2: Run as Script

```bash
python -m src.model
```
