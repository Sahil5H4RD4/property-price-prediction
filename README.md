# Intelligent Property Price Prediction System

An AI-driven real estate analytics system that predicts property prices using historical data and machine learning models. Built with scikit-learn, Streamlit, and Plotly.

> **Milestone 1** — ML-Based Property Price Prediction

---

## Problem Statement

Real estate pricing is complex and influenced by numerous factors — property size, location features, amenities, and market conditions. This project aims to:

1. **Predict property prices** using supervised machine learning models
2. **Identify key price-driving factors** through feature importance analysis
3. **Provide an interactive UI** for users to input property details and get instant predictions
4. **Support batch predictions** via CSV upload

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB UI                          │
│  ┌──────────────┐ ┌───────────────┐ ┌────────────────────┐  │
│  │ Property Form│ │ Model Insights│ │   Batch Upload     │  │
│  │  (Input)     │ │  (Charts)     │ │   (CSV → Pred)     │  │
│  └──────┬───────┘ └───────┬───────┘ └────────┬───────────┘  │
│         │                 │                   │              │
├─────────┼─────────────────┼───────────────────┼──────────────┤
│         ▼                 ▼                   ▼              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │              PREDICTION ENGINE (predict.py)          │     │
│  │   • Load trained model  • Preprocess input           │     │
│  │   • Generate predictions • Batch processing          │     │
│  └──────────────────────┬──────────────────────────────┘     │
│                         │                                    │
│  ┌──────────────────────┼──────────────────────────────┐     │
│  │        PREPROCESSING PIPELINE (preprocess.py)        │     │
│  │   • Binary encoding    • One-hot encoding            │     │
│  │   • Feature engineering • StandardScaler             │     │
│  └──────────────────────┬──────────────────────────────┘     │
│                         │                                    │
│  ┌──────────────────────┼──────────────────────────────┐     │
│  │          ML MODELS (train.py)                        │     │
│  │   • Linear Regression  • Decision Tree               │     │
│  │   • Random Forest      • Gradient Boosting           │     │
│  └──────────────────────┬──────────────────────────────┘     │
│                         │                                    │
│  ┌──────────────────────▼──────────────────────────────┐     │
│  │            DATASET (Housing.csv)                     │     │
│  │   545 properties × 13 features                       │     │
│  └─────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset

**Source:** [Kaggle Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

| Feature | Type | Description |
|---------|------|-------------|
| `price` | Numeric | Target variable — property price |
| `area` | Numeric | Property area in sq ft |
| `bedrooms` | Numeric | Number of bedrooms (1–6) |
| `bathrooms` | Numeric | Number of bathrooms (1–4) |
| `stories` | Numeric | Number of stories (1–4) |
| `mainroad` | Binary | Main road access (yes/no) |
| `guestroom` | Binary | Guest room available (yes/no) |
| `basement` | Binary | Basement available (yes/no) |
| `hotwaterheating` | Binary | Hot water heating (yes/no) |
| `airconditioning` | Binary | Air conditioning (yes/no) |
| `parking` | Numeric | Number of parking spots (0–3) |
| `prefarea` | Binary | Preferred area (yes/no) |
| `furnishingstatus` | Categorical | furnished / semi-furnished / unfurnished |

**Records:** 545 properties | **Features:** 13 (12 input + 1 target)

---

## Feature Engineering

Starting from 12 raw features, the preprocessing pipeline creates **18 features**:

| Engineered Feature | Description |
|-------------------|-------------|
| `total_rooms` | bedrooms + bathrooms |
| `area_per_room` | area / total_rooms |
| `amenity_score` | Sum of all binary amenities |
| `area_tier` | Property size tier (0/1/2 based on quantiles) |

All binary columns are encoded as 0/1, `furnishingstatus` is one-hot encoded, and all features are StandardScaled.

---

## Model Performance

| Model | MAE (₹) | RMSE (₹) | R² Score |
|-------|---------|----------|----------|
| **Linear Regression** | **430,716** | **531,133** | **0.9390** |
| Gradient Boosting | 605,993 | 764,516 | 0.8736 |
| Random Forest | 734,138 | 903,235 | 0.8236 |
| Decision Tree | 1,139,850 | 1,462,994 | 0.5373 |

**Best Model:** Linear Regression with R² = 0.9390

### Top Price-Driving Factors
1. **Area** — Most influential factor
2. **Stories** — Number of floors
3. **Total Rooms** — Combined bedrooms + bathrooms
4. **Bathrooms** — Individual bathroom count
5. **Parking** — Number of parking spots

---

## User Interface

The Streamlit app provides three main views:

### Predict Price
- Input property details via form (area, bedrooms, amenities, etc.)
- Get instant price prediction with model confidence
- See price per sq ft and estimated EMI

### Model Insights
- R² score comparison across all trained models
- MAE comparison chart
- Detailed metrics table
- Feature importance horizontal bar chart

### Batch Upload
- Upload CSV file with multiple properties
- Get predictions for all rows
- View distribution of predicted prices
- Download results as CSV

---

## Quick Start

### Prerequisites
- Python 3.9+

### Installation

```bash
# Clone the repository
git clone https://github.com/Sahil5H4RD4/property-price-prediction.git
cd property-price-prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Train the Model
```bash
python src/train.py
```

### Launch the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
property-price-prediction/
├── .streamlit/
│   └── config.toml          # Streamlit theme configuration
├── data/
│   ├── Housing.csv           # Dataset (545 rows)
│   └── eda_plots/            # EDA visualization outputs
├── models/
│   ├── best_model.pkl        # Trained best model (Linear Regression)
│   ├── scaler.pkl            # Fitted StandardScaler
│   ├── feature_names.pkl     # Feature name list
│   ├── feature_importance.csv
│   └── model_info.json       # Model metadata & metrics
├── notebooks/
│   └── eda.py                # Exploratory Data Analysis script
├── src/
│   ├── __init__.py
│   ├── preprocess.py         # Data preprocessing & feature engineering
│   ├── train.py              # Model training & evaluation pipeline
│   └── predict.py            # Prediction utilities
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## Tech Stack

| Category | Tool |
|----------|------|
| ML | scikit-learn |
| Data Processing | pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| UI Framework | Streamlit |
| Language | Python 3.12 |

---

## License

This project is for educational purposes as part of the AI/ML course project.
