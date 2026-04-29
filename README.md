# Apple Stock Price Prediction

A machine learning project that predicts the direction of Apple (AAPL) stock price movement using historical daily trading data from 2010 to 2020.

## Dataset

`Apple_Stock.csv` contains 2,518 trading days with the following fields:

| Column | Description |
|---|---|
| Date | Trading date (MM/DD/YYYY) |
| Close/Last | Closing price |
| Volume | Number of shares traded |
| Open | Opening price |
| High | Daily high price |
| Low | Daily low price |

## Approach

**Target:** Binary classification — whether the next day's closing price will be higher (1) or lower (0) than today's.

**Engineered features:**
- `Open-Last` — difference between open and close price
- `Low-High` — difference between daily low and high (spread)
- `Quarter_end` — flag for quarter-end months (March, June, September, December)

Features are standardized with `StandardScaler` before training.

**Models evaluated:**
- Logistic Regression
- Support Vector Classifier (polynomial kernel)

## Results

| Model | Train AUC | Validation AUC |
|---|---|---|
| Logistic Regression | 0.533 | 0.465 |
| SVC (poly kernel) | 0.538 | 0.472 |

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Usage

Open `Apple_Stock.ipynb` in Jupyter and run all cells.
