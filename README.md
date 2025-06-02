# Momentum Trading Screener

A Python-based momentum trading screener that identifies high-momentum stocks using both simple and enhanced momentum calculation methods. The screener analyzes historical price data to generate momentum scores and select top-performing stocks for portfolio construction.

## Features

- **Dual Momentum Calculation Methods**:
  - Simple momentum: Basic price momentum with volatility adjustment
  - Enhanced momentum: Multi-factor approach including:
    - Short-term (1 month)
    - Medium-term (6 months)
    - Long-term (12 months) momentum
    - Volatility adjustment
    - Trend strength analysis

- **Portfolio Analysis**:
  - Equal-weighted portfolio simulation
  - Benchmark comparison against SPY
  - Performance visualization
  - Historical comparison with previous periods

- **Data Management**:
  - Automated data download using yfinance
  - Retry logic for failed downloads
  - Efficient data processing and storage
  - CSV export of results

## Requirements

- Python 3.7+
- Required packages:
  - yfinance
  - pandas
  - numpy
  - matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/momentum.git
cd momentum
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your ticker list:
   - Create a text file (e.g., `cedears.txt`) with one ticker symbol per line
   - Ensure tickers are in the correct format (e.g., 'AAPL' for Apple)

2. Run the screener:
```bash
python screener.py
```

3. View the results:
   - Top momentum stocks will be displayed in the console
   - A performance comparison chart will be shown
   - Results will be saved to CSV files with timestamps

## Configuration

The screener can be configured through the `MomentumScreener` class parameters:

```python
screener = MomentumScreener(
    n_stocks=10,              # Number of top stocks to select
    lookback_days=252,        # Days to look back for momentum calculation
    gap_days=21,             # Days to exclude from the end
    max_retries=3,           # Maximum download retry attempts
    timeout=30,              # Download timeout in seconds
    momentum_method='simple' # 'simple' or 'enhanced'
)
```

## Output Files

The screener generates two types of output files:

1. `momentum_portfolio_[timestamp].csv`:
   - Contains the selected stocks and their momentum scores
   - Includes factor exposures for enhanced momentum method

2. `momentum_comparison_[timestamp].csv`:
   - Compares current results with previous month
   - Shows added, removed, and maintained stocks

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and research purposes only. It should not be used as financial advice. Always do your own research before making investment decisions. 