# Finance & Economics Time-Series Analysis

An end-to-end data science project analyzing how macroeconomic indicators such as GDP growth, inflation, interest rates, and unemployment interact with US stock market performance (S&P 500) between **2000–2008**.

The project combines **classical time-series models (ARIMA, VAR)** with **deep learning (LSTM)** and wraps everything in an interactive **Streamlit dashboard**.

---

### Instructions to run the project:

(Recommended) Use a virtual environment to run the project.

**Install the Libraries:**
  - pip install -r requirements.txt
  
**Run the Dashboard:**
  - streamlit run finance_app_final.py

## Project Overview

**Problem Statement**

Financial markets and the broader economy are tightly linked. GDP growth, inflation, interest rates, unemployment, and stock prices all influence one another in complex, time-dependent ways.  

The goal of this project is to:

- Explore and clean a combined **Finance & Economics Dataset**.
- Engineer meaningful financial and macroeconomic features.
- Build and compare multiple forecasting models for the S&P 500.
- Study how macroeconomic shocks (e.g., unemployment changes) relate to stock market movements.
- Present all results in an easy-to-use interactive dashboard.

---

## Dataset

**Name:** `Finance & Economics Dataset`  
**Granularity:** Daily data (2000–2008)  

Includes:

- **Market Data (S&P 500 focus)**
  - Open, High, Low, Close, Trading Volume
- **Macroeconomic Indicators**
  - GDP Growth (%)
  - Inflation Rate (%)
  - Interest Rate (%)
  - Unemployment Rate (%)
  - Consumer Confidence Index
- Additional derived/engineered features (moving averages, lagged variables, RSI, etc.) are created in the project.

**Name:** `US Recession Dataset`
**Granularity:** 1990-04 to 2022-10

---

## What This Project Implements

### 1. Dataset Exploration & Cleaning
- Filter raw dataset to focus on **S&P 500** only (removes mixed index noise).
- Convert date column to a proper datetime index and ensure continuous time-series.
- Handle missing values using forward/backward fill and type conversion.
- Restrict the analysis window to **2000-01-01 to 2008-12-31**.

### 2. Exploratory Data Analysis (EDA)
- Time-series plots for key indicators (S&P 500, GDP, inflation, unemployment).
- **Seasonal decomposition** (trend, seasonality, residuals) for the S&P 500.
- **Correlation heatmap** to quantify relationships between macro variables and stock prices.
- Normalized index comparison (S&P vs other indices) to justify focusing on S&P 500.

### 3. Feature Engineering
- Rolling features:
  - 90-day & 365-day moving averages of S&P 500.
- Lag features:
  - Lagged GDP (e.g., 90-day lag).
- Returns & transforms:
  - Log returns for S&P 500.
- Technical indicators:
  - **RSI (Relative Strength Index)** for momentum.
- All engineered features are visualized and inspected.

### 4. Forecasting Models

**Univariate Model – ARIMA**
- Classical ARIMA model on S&P 500 prices.
- Stationarity check using **ADF test**.
- User-controlled `(p, d, q)` parameters via the UI.
- Forecast vs actual plots with RMSE / MAE.

**Multivariate Model – VAR**
- Vector Autoregression on:
  - `SP500_Price`, `GDP`, `UnemploymentRate`
- Safe lag selection using information criteria.
- Forecast of S&P 500 incorporating macroeconomic variables.
- Visualization comparing actual vs VAR forecast.

**Deep Learning – LSTM**
- Single-step forecasting of S&P 500 using:
  - MinMax-scaled prices.
  - 30-day sliding window as sequence input.
- Custom PyTorch LSTM implementation.
- Visual comparison of actual vs LSTM predictions.

**All-Model Comparison**
- ARIMA vs VAR vs LSTM on the same test window.
- Metrics:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
- Side-by-side comparison plot to see which model tracks the 2008 crisis best.

### 5. Sentiment (Educational Simulation)
- Simulated sentiment score derived from S&P 500 returns plus noise.
- Time-series visualization to demonstrate how sentiment might precede volatility.
- Explanation of how a real system would plug in a BERT-style NLP model on news/headlines.

### 6. Conclusions & Insights
- Justification for choosing S&P 500 as the main market proxy.
- Comparison of ARIMA vs LSTM vs VAR in terms of capturing crises vs normal periods.
- High-level takeaway on how macroeconomic indicators (e.g., unemployment) relate to stock movements.

---

## Interactive Dashboard Structure

The Streamlit app (`finance_app_final.py`) is organized into the following pages:

1. **Dataset & Cleaning**
   - Preview of cleaned S&P 500 + macro dataset.
   - Explanation of filtering and cleaning logic.

2. **EDA & Comparative Analysis**
   - Index comparison (normalized performance).
   - Seasonal decomposition of S&P 500.
   - Correlation heatmap and interpretation.

3. **Feature Engineering**
   - Display of engineered features (MA, log returns, RSI, lagged GDP).
   - Visualizations of moving averages and RSI.

4. **Forecasting Models (ARIMA/VAR/LSTM)**
   - Baseline ARIMA model (univariate).
   - Multivariate VAR model.
   - Deep learning LSTM model.
   - Combined comparison of ARIMA vs VAR vs LSTM.

5. **Sentiment Analysis (Educational Simulation)**
   - Simulated sentiment index vs time.
   - Narrative on how real sentiment models would integrate.

6. **Conclusions**
   - Summary of methods, findings, limitations, and future directions.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Dashboard:** Streamlit
- **Time-Series / Stats:** `statsmodels` (ARIMA, VAR), `numpy`, `pandas`
- **Deep Learning:** PyTorch (custom LSTM)
- **Visualization:** Matplotlib, Seaborn, Streamlit native charts
- **Other:** scikit-learn (scaling, metrics)

All Python dependencies are listed in `requirements.txt`.
