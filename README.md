# Financial Market Volatility & Risk Dashboard

An interactive Python and Streamlit application for analyzing historical stock
performance, volatility, downside risk, and benchmark relationships using live
Yahoo Finance data.

## Live Demo

[Launch the Financial Market Volatility & Risk Dashboard](https://financial-risk-dashboard-sylvia.streamlit.app)

## Live project goal

Instead of using a hard-coded dataset, the dashboard lets a user:

- Enter a stock ticker such as `AAPL`, `NVDA`, `DIS`, or `NFLX`
- Choose a custom start and end date
- Select a market benchmark
- Set the risk-free rate used for the Sharpe ratio
- Run the analysis on demand
- Explore interactive performance and risk charts
- Download the calculated analysis data as CSV

## Metrics calculated

The application calculates:

- Starting and ending price
- Total return
- Average daily return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- 95% historical Value at Risk (VaR)
- 20-day moving average
- 50-day moving average
- 30-day rolling annualized volatility
- Correlation with a selected market benchmark

The dashboard also assigns a simple educational volatility-based risk
classification: **Low Risk**, **Moderate Risk**, or **High Risk**.

## Visualizations

The app includes:

1. Historical adjusted closing price
2. 20-day and 50-day moving averages
3. Daily return chart
4. Rolling 30-day volatility
5. Drawdown chart
6. Daily-return distribution with the 95% VaR threshold
7. Growth-of-$100 comparison against a market benchmark

Plotly is used for interactive charts and Matplotlib is used for the historical
VaR return-distribution visualization.

## Technology

- Python
- Streamlit
- pandas
- NumPy
- yfinance
- Plotly
- Matplotlib

## Project structure

```text
financial-market-risk-dashboard/
├── app.py
├── analysis.py
├── requirements.txt
├── README.md
└── .gitignore
```

`analysis.py` contains the market-data and financial-calculation functions.

`app.py` contains the Streamlit interface, user controls, metric cards,
visualizations, tabs, and CSV export.

## Run locally

From the project folder:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Then open the local Streamlit URL shown in Terminal.

## Deploy

This project is designed for Streamlit Community Cloud.

Use:

- Repository: your GitHub repository
- Branch: `main`
- Main file: `app.py`

## Important note

This project is for educational and portfolio purposes only. It does not provide
financial advice, and its simplified risk classification should not be used to
make investment decisions.
