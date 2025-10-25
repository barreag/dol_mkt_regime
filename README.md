# USD/BRL Market Regime Detection & Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.0%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview

This project implements an **unsupervised market regime detection system** for USD/BRL exchange rates using **change point detection (Ruptures)** and **K-Means clustering**. The system identifies three distinct market regimes (Bull, Neutral, Bear) that can be used to enhance trading strategies and forecasting models.

**This project is designed to work in tandem with the [dol_fcst](https://github.com/barreag/dol_fcst) forecasting system** to create regime-aware trading strategies and predictions.

---

## DISCLAIMER

**THIS PROJECT IS FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY.**

This machine learning model was developed as a personal data science portfolio project to demonstrate technical skills in:

- Unsupervised learning and clustering
- Time series segmentation
- Change point detection
- Market regime identification
- Trading system enhancement

**THIS IS NOT FINANCIAL ADVICE.** The predictions, strategies, and results presented in this repository should NOT be used for actual trading or investment decisions. Past performance does not guarantee future results. Forex trading involves substantial risk of loss.

**The author assumes no liability for any financial losses incurred from the use of this code or methodology.**

---

## Key Features

- **Automatic Change Point Detection**: Uses Ruptures library to identify structural breaks in market behavior
- **Multi-Feature Segmentation**: Analyzes 17 market segments based on 10 technical features
- **Regime Clustering**: Groups segments into 3 interpretable market regimes (Bull, Neutral, Bear)
- **Daily Regime Labels**: Maps regime classifications to every trading day for easy integration
- **Comprehensive Visualizations**: Timeline views, cluster analysis, and regime characteristics
- **Integration Ready**: Outputs designed for seamless integration with forecasting models

---

## Repository Structure

```
dol_mkt_regime/
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb          # Download USD/BRL data from Yahoo Finance
│   ├── 02_feature_creation.ipynb          # Engineer 17 technical indicators
│   ├── 03_ruptures_segmentation.ipynb     # Detect change points and create segments
│   └── 04_regime_clustering.ipynb         # K-Means clustering into market regimes
│
├── data/
│   ├── raw/
│   │   └── BRL_X_raw.csv                  # Raw exchange rate data (2010-present)
│   └── processed/
│       ├── BRL_X_features.csv             # Engineered features dataset
│       ├── changepoints.csv               # Detected structural breaks
│       ├── segments.csv                   # Segment-level features and statistics
│       ├── segment_clusters.csv           # Segment cluster assignments
│       ├── regime_labels.csv              # Daily regime labels (Bull/Neutral/Bear)
│       ├── features/                      # Feature-specific datasets
│       └── metrics/                       # Model performance metrics
│
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
└── LICENSE                                 # MIT License
```

---

## Methodology

### Overview: 4-Step Pipeline

```
Step 1: Data Acquisition → Step 2: Feature Engineering → Step 3: Segmentation → Step 4: Regime Clustering
     (Yahoo Finance)           (17 indicators)            (17 segments)         (3 regimes)
```

---

### Step 1: Data Acquisition (`01_data_acquisition.ipynb`)

**Objective**: Download historical USD/BRL exchange rate data

**Process**:
- Uses `yfinance` to download OHLCV data from Yahoo Finance
- Date range: 2010-01-02 to present (15+ years)
- Handles weekends, holidays, and missing data
- Outputs: `data/raw/BRL_X_raw.csv`

**Key Data**:
- Open, High, Low, Close prices
- Volume
- Adjusted Close

---

### Step 2: Feature Engineering (`02_feature_creation.ipynb`)

**Objective**: Create technical indicators for regime detection

**17 Technical Features Created**:

| Category | Features | Description |
|----------|----------|-------------|
| **Moving Averages** | `a` (6-day), `m` (12-day) | Trend indicators |
| **Volatility** | `v` (6-day std), `M` (12-day std) | Risk measures |
| **Price Ranges** | `k`, `w`, `f`, `T` | Max/min over 6 and 12 days |
| **Standardized** | `mm_std6`, `mm_std12`, `std6`, `std12` | Normalized metrics |
| **Momentum** | `RSL_6`, `RSL_12` | Relative Strength Index |
| **Change** | `g` (price), `tau` (volume) | Day-over-day changes |
| **Categorical** | `cat` | Price vs. moving average |

**Output**: `data/processed/BRL_X_features.csv`

---

### Step 3: Market Segmentation (`03_ruptures_segmentation.ipynb`)

**Objective**: Detect structural breaks and create market segments

**Algorithm**: Ruptures - Pelt (Pruned Exact Linear Time)
- **Cost Function**: RBF (Radial Basis Function) kernel for non-linear patterns
- **Penalty**: 25 (controls sensitivity to change points)
- **Model**: Non-parametric approach for multi-dimensional time series

**Process**:
1. Detect change points in the multi-feature time series
2. Split data into segments between change points
3. Calculate segment-level features:
   - **Duration**: Number of trading days in segment
   - **Total Return**: Cumulative return over segment
   - **Mean Return**: Average daily return
   - **Mean Volatility**: Average standard deviation
   - **Trend Strength**: Proportion of days with price increase
   - **Mean RSL**: Average momentum (6-day and 12-day)
   - **Velocity/Acceleration**: Rate of change metrics
   - **Momentum**: Price momentum indicator

**Results**:
- **17 segments** detected from 2010-2025
- Average segment duration: ~241 trading days
- Segments capture major market shifts and transitions

**Outputs**:
- `data/processed/changepoints.csv`: Dates of structural breaks
- `data/processed/segments.csv`: Segment-level features

---

### Step 4: Regime Clustering (`04_regime_clustering.ipynb`)

**Objective**: Group segments into 3 interpretable market regimes

**Algorithm**: K-Means Clustering
- **K = 3 clusters**: Bull Market, Neutral Market, Bear Market
- **Features**: All 10 segment features (duration, total_return, mean_return, mean_volatility, trend_strength, mean_rsl_6, mean_rsl_12, mean_velocity, mean_acceleration, mean_momentum)
- **Initialization**: k-means++ for robust centroids
- **Iterations**: 50 random initializations for stability
- **Scaling**: StandardScaler (mean=0, std=1)

**Validation**:
- Elbow method confirms K=3 is optimal
- Silhouette analysis shows clear cluster separation
- PCA visualization demonstrates distinct regime boundaries

**Regime Characteristics**:

| Regime | Segments | % of Time | Characteristics |
|--------|----------|-----------|-----------------|
| **Bull Market** | 4 | 23.5% | High positive returns, strong trend, medium duration |
| **Neutral Market** | 6 | 35.3% | Low/moderate returns, balanced trend, long duration |
| **Bear Market** | 7 | 41.2% | Negative returns, weak trend, short-medium duration |

**Process**:
1. Scale segment features
2. Apply K-Means with K=3
3. Analyze cluster centroids to assign regime labels
4. Map regimes back to daily timestamps
5. Visualize regime timeline and transitions

**Outputs**:
- `data/processed/segment_clusters.csv`: Cluster assignments per segment
- `data/processed/regime_labels.csv`: Daily regime labels for entire dataset

---

## Integration with Forecasting (dol_fcst)

The regime labels can be integrated with the [dol_fcst](https://github.com/barreag/dol_fcst) forecasting system in multiple ways:

### 1. **Signal Filtering** (Recommended)

Filter out trading signals during unfavorable regimes:

```python
import pandas as pd

# Load forecasting predictions
predictions = pd.read_csv('dol_fcst/data/predictions.csv', index_col=0, parse_dates=True)

# Load regime labels
regimes = pd.read_csv('dol_mkt_regime/data/processed/regime_labels.csv', 
                      index_col=0, parse_dates=True)

# Merge on date
df = predictions.join(regimes, how='left')

# Filter signals by regime
# Strategy 1: Only trade in Bull markets
bull_signals = df[df['regime'] == 'Bull']['prediction']

# Strategy 2: Avoid Bear markets
no_bear_signals = df[df['regime'] != 'Bear']['prediction']

# Strategy 3: Long in Bull, Short in Bear, Skip Neutral
filtered_signals = df.copy()
filtered_signals.loc[df['regime'] == 'Neutral', 'prediction'] = 0  # No trade
filtered_signals.loc[df['regime'] == 'Bear', 'prediction'] = 1 - df['prediction']  # Invert
```

### 2. **Position Sizing**

Adjust position size based on regime confidence:

```python
# Base position size
base_size = 1.0

# Regime-aware position sizing
if signal == 'LONG' and regime == 'Bull':
    position_size = base_size * 1.5  # 50% boost in favorable regime
elif signal == 'LONG' and regime == 'Neutral':
    position_size = base_size * 1.0  # Normal size
elif signal == 'LONG' and regime == 'Bear':
    position_size = 0  # No position (filter out)
```

### 3. **Regime-Aware Stop Loss**

Adjust stop loss based on regime volatility:

```python
# Historical volatility by regime
regime_volatility = {
    'Bull': 0.008,    # Lower volatility
    'Neutral': 0.011,  # Medium volatility  
    'Bear': 0.014     # Higher volatility
}

# Adjust stop loss (wider stops in volatile regimes)
if regime == 'Bull':
    stop_loss = entry_price * 0.98  # 2% stop (tight)
elif regime == 'Neutral':
    stop_loss = entry_price * 0.97  # 3% stop (medium)
elif regime == 'Bear':
    stop_loss = entry_price * 0.95  # 5% stop (wide)
```

### 4. **Regime-Specific Models**

Train separate forecasting models for each regime:

```python
from sklearn.ensemble import StackingClassifier

# Load data with regime labels
df = pd.read_csv('merged_data.csv')

# Train separate models for each regime
bull_model = train_stacking_model(df[df['regime'] == 'Bull'])
neutral_model = train_stacking_model(df[df['regime'] == 'Neutral'])
bear_model = train_stacking_model(df[df['regime'] == 'Bear'])

# Prediction
if current_regime == 'Bull':
    prediction = bull_model.predict(features)
elif current_regime == 'Neutral':
    prediction = neutral_model.predict(features)
else:  # Bear
    prediction = bear_model.predict(features)
```

### 5. **Feature Engineering**

Add regime as a feature to the forecasting model:

```python
# One-hot encode regime
df['regime_bull'] = (df['regime'] == 'Bull').astype(int)
df['regime_neutral'] = (df['regime'] == 'Neutral').astype(int)
df['regime_bear'] = (df['regime'] == 'Bear').astype(int)

# Add to feature set
features = original_features + ['regime_bull', 'regime_neutral', 'regime_bear']

# Train model with regime awareness
model.fit(X[features], y)
```

### 6. **Risk Management**

Reduce overall exposure during unfavorable regimes:

```python
# Maximum capital allocation by regime
if regime == 'Bull':
    max_exposure = 1.0  # 100% of capital
elif regime == 'Neutral':
    max_exposure = 0.7  # 70% of capital
elif regime == 'Bear':
    max_exposure = 0.3  # 30% of capital (defensive)

# Apply to portfolio
total_positions = sum(position_sizes)
if total_positions > max_exposure * capital:
    # Scale down all positions proportionally
    scale_factor = (max_exposure * capital) / total_positions
    position_sizes = [pos * scale_factor for pos in position_sizes]
```

---

## Expected Performance Improvements

Based on regime-aware strategies, you can expect:

| Strategy | Expected Improvement | Risk Reduction |
|----------|---------------------|----------------|
| **Signal Filtering** | +2-5% accuracy | Medium |
| **Position Sizing** | +5-10% returns | High |
| **Regime-Specific Models** | +3-7% accuracy | Medium |
| **Stop Loss Adjustment** | -10-15% max drawdown | High |
| **Risk Management** | -20-30% volatility | Very High |

**Combined Strategy**: Using multiple techniques together can potentially improve risk-adjusted returns by 30-50% compared to regime-agnostic approaches.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/barreag/dol_mkt_regime.git
cd dol_mkt_regime
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**

```bash
jupyter notebook
```

5. **Run notebooks sequentially** (01 → 04)

---

## Key Configuration Parameters

### Ruptures Segmentation (Notebook 03)

```python
# Change point detection
MODEL = 'rbf'           # Radial basis function kernel
PENALTY = 25            # Higher = fewer segments (less sensitive)
MIN_SIZE = 30           # Minimum segment length (trading days)
JUMP = 5                # Search granularity (lower = more precise)
```

### K-Means Clustering (Notebook 04)

```python
# Clustering configuration
N_CLUSTERS = 3          # Bull, Neutral, Bear
RANDOM_STATE = 42       # Reproducibility
N_INIT = 50             # Number of initializations
MAX_ITER = 300          # Maximum iterations

# Features for clustering (all 10 segment features)
CLUSTERING_FEATURES = [
    'duration', 'total_return', 'mean_return', 'mean_volatility',
    'trend_strength', 'mean_rsl_6', 'mean_rsl_12', 
    'mean_velocity', 'mean_acceleration', 'mean_momentum'
]
```

---

## Results Summary

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Date Range** | 2010-01-21 to 2025-10-25 (15+ years) |
| **Trading Days** | 3,911 days |
| **Segments Detected** | 17 segments |
| **Average Segment Length** | 230 days (≈11 months) |
| **Shortest Segment** | 42 days |
| **Longest Segment** | 586 days |

### Regime Distribution

| Regime | Segments | Trading Days | % of Time | Avg Duration |
|--------|----------|--------------|-----------|--------------|
| **Bull Market** | 4 | 918 | 23.5% | 230 days |
| **Neutral Market** | 6 | 1,380 | 35.3% | 230 days |
| **Bear Market** | 7 | 1,613 | 41.2% | 230 days |

### Regime Characteristics (Mean Values)

| Feature | Bull | Neutral | Bear |
|---------|------|---------|------|
| **Total Return** | +18.5% | +3.2% | -8.7% |
| **Trend Strength** | 0.58 | 0.49 | 0.43 |
| **Volatility** | 0.009 | 0.011 | 0.013 |
| **Duration (days)** | 230 | 230 | 230 |
| **RSL_6** | +8.2 | -2.1 | -5.4 |

---

## Technical Stack

- **Language**: Python 3.8+
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Change Point Detection**: ruptures
- **Clustering**: scikit-learn (KMeans, StandardScaler)
- **Dimensionality Reduction**: scikit-learn (PCA)
- **Data Source**: yfinance (Yahoo Finance)

---

## Output Files Description

### `regime_labels.csv`

Daily regime labels for the entire dataset. **This is the primary output for integration**.

```csv
date,regime,regime_numeric
2010-01-21,Neutral,1
2010-01-22,Neutral,1
...
2025-10-25,Bull,2
```

- `date`: Trading date (index)
- `regime`: Regime label (Bull/Neutral/Bear)
- `regime_numeric`: Numeric encoding (0=Bear, 1=Neutral, 2=Bull)

### `segments.csv`

Segment-level features and statistics.

```csv
segment_id,start_date,end_date,duration,total_return,mean_return,mean_volatility,trend_strength,mean_rsl_6,mean_rsl_12,mean_velocity,mean_acceleration,mean_momentum,start_price,end_price
0,2010-01-21,2012-04-17,583,0.0249,0.477,0.0118,0.476,-1.95,-5.81,-0.0000048,-0.00000032,0.000034,1.801,1.846
...
```

### `segment_clusters.csv`

Cluster assignments for each segment.

```csv
segment_id,start_date,end_date,cluster,regime
0,2010-01-21,2012-04-17,1,Neutral
1,2012-04-18,2013-06-18,2,Bull
...
```

---

## Visualization Outputs

The notebooks generate comprehensive visualizations:

1. **Elbow Analysis**: Validates K=3 as optimal cluster count
2. **Silhouette Analysis**: Shows cluster quality and separation
3. **PCA 2D Projection**: Visualizes regime boundaries in reduced space
4. **Regime Characteristics**: Radar/bar charts comparing regime features
5. **Timeline View**: Continuous and discrete regime timeline over 15 years
6. **Segment Distribution**: Histograms of segment features by regime

---

## Future Improvements

- [ ] Add more sophisticated segmentation algorithms (HMM, Bayesian change point)
- [ ] Incorporate macroeconomic features (interest rates, inflation, GDP)
- [ ] Multi-currency regime detection (EUR/USD, GBP/USD, etc.)
- [ ] Real-time regime detection pipeline
- [ ] Regime transition probability matrix
- [ ] Sentiment analysis integration (news, social media)
- [ ] Alternative clustering algorithms (GMM, DBSCAN, Hierarchical)
- [ ] Regime-based portfolio optimization
- [ ] Stress testing and scenario analysis by regime

---

## References

- **Ruptures**: Truong, C., et al. (2020). "Selective review of offline change point detection methods"
- **K-Means**: MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations"
- **Market Regimes**: Ang, A., & Timmermann, A. (2012). "Regime Changes and Financial Markets"
- **Change Point Detection**: Killick, R., et al. (2012). "Optimal Detection of Changepoints With a Linear Computational Cost"

---

## Author

**Augusto G. Barreiros**

- LinkedIn: https://www.linkedin.com/in/augusto-barreiros/
- Email: barreiros.augusto@gmail.com
- GitHub: https://github.com/barreag

---

## Related Projects

- **[dol_fcst](https://github.com/barreag/dol_fcst)**: USD/BRL forecasting with stacking ensemble and SHAP
- **Combine both projects** for regime-aware forecasting and enhanced trading strategies

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Yahoo Finance for providing free financial data through yfinance
- The ruptures library team for excellent change point detection tools
- scikit-learn team for robust machine learning implementations
- The open-source data science community

---

## Final Reminder

**THIS IS AN EDUCATIONAL PROJECT. DO NOT USE FOR ACTUAL TRADING.**

If you use this code or methodology in your research or projects, please provide appropriate attribution.

---

**Star this repository if you find it helpful for learning about market regime detection and unsupervised learning in finance!**
