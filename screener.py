import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional

class MomentumScreener:
    def __init__(self, 
                 n_stocks: int = 10,
                 lookback_days: int = 252,
                 gap_days: int = 21,
                 max_retries: int = 3,
                 timeout: int = 30,
                 momentum_method: str = 'simple'):
        """
        Initialize the momentum screener with configuration parameters.
        
        Args:
            n_stocks: Number of top stocks to select
            lookback_days: Number of days to look back for momentum calculation
            gap_days: Number of days to exclude from the end (to avoid look-ahead bias)
            max_retries: Maximum number of download retry attempts
            timeout: Timeout in seconds for data downloads
            momentum_method: 'simple' or 'enhanced' momentum calculation
        """
        self.n_stocks = n_stocks
        self.lookback_days = lookback_days
        self.gap_days = gap_days
        self.max_retries = max_retries
        self.timeout = timeout
        self.momentum_method = momentum_method.lower()
        
        # Initialize data storage
        self.price_data = pd.DataFrame()
        self.scores = None
        self.top_stocks = None
        self.portfolio_value = None
        self.spy_returns = None

    def load_tickers(self, filename: str) -> List[str]:
        """Load ticker symbols from a file."""
        try:
            with open(filename, 'r') as f:
                # Read lines and filter out empty lines and whitespace
                tickers = [line.strip() for line in f.readlines() if line.strip()]
            # Replace dots with hyphens for yfinance compatibility
            return [ticker.replace('.', '-') for ticker in tickers]
        except FileNotFoundError:
            print(f"Error: File {filename} not found")
            return []
        except Exception as e:
            print(f"Error reading file {filename}: {str(e)}")
            return []

    def download_data(self, tickers: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Download price data for a list of tickers with retry logic."""
        successful_downloads = 0
        downloaded_data = []

        for ticker in tickers:
            for attempt in range(self.max_retries):
                try:
                    ticker_data = yf.download(
                        ticker, 
                        start=start_date, 
                        end=end_date, 
                        progress=False, 
                        timeout=self.timeout
                    )['Close']
                    
                    if not ticker_data.empty:
                        ticker_data.name = ticker  # Set the name for the Series
                        downloaded_data.append(ticker_data)
                        successful_downloads += 1
                        print(f"\n✓ Downloaded {ticker}")
                        break
                    else:
                        print(f"\n✗ No data available for {ticker}")
                        break
                        
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        print(f"! Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                        time.sleep(2)
                    else:
                        print(f"✗ Failed to download {ticker} after {self.max_retries} attempts")

        print(f"\nSuccessfully downloaded data for {successful_downloads} out of {len(tickers)} tickers")
        
        if downloaded_data:
            # Combine all downloaded data at once using pd.concat
            data = pd.concat(downloaded_data, axis=1)
            return data.dropna(axis=1)
        else:
            return pd.DataFrame()

    def calculate_simple_momentum(self) -> pd.Series:
        """Calculate simple momentum scores using the original method."""
        returns = self.price_data.pct_change()
        momentum_return = self.price_data.shift(self.gap_days).pct_change(self.lookback_days - self.gap_days)
        volatility = returns.rolling(window=self.lookback_days).std().iloc[-1]
        
        self.scores = (momentum_return.iloc[-1] / volatility).dropna()
        return self.scores

    def calculate_enhanced_momentum(self) -> pd.Series:
        """
        Calculate enhanced momentum scores using multiple factors:
        1. Price momentum (short, medium, long-term)
        2. Volatility adjustment
        3. Volume momentum
        4. Trend strength
        """
        # Calculate returns for different periods
        returns = self.price_data.pct_change()
        
        # Short-term momentum (1 month)
        short_term = returns.rolling(window=21).sum().iloc[-1]
        
        # Medium-term momentum (6 months)
        medium_term = returns.rolling(window=126).sum().iloc[-1]
        
        # Long-term momentum (12 months, excluding last month)
        long_term = self.price_data.shift(self.gap_days).pct_change(self.lookback_days - self.gap_days).iloc[-1]
        
        # Calculate volatility (using exponential weighted moving average for more recent emphasis)
        volatility = returns.ewm(span=self.lookback_days).std().iloc[-1]
        
        # Calculate trend strength using ADX-like measure
        high_low = self.price_data.rolling(window=14).max() - self.price_data.rolling(window=14).min()
        trend_strength = (high_low / self.price_data.rolling(window=14).mean()).iloc[-1]
        
        # Combine factors with weights
        momentum_score = (
            0.4 * long_term +    # Long-term momentum (40% weight)
            0.3 * medium_term +  # Medium-term momentum (30% weight)
            0.3 * short_term     # Short-term momentum (30% weight)
        )
        
        # Risk adjustment
        risk_adjusted_score = momentum_score / volatility
        
        # Apply trend strength filter
        final_score = risk_adjusted_score * trend_strength
        
        # Normalize scores to have mean 0 and std 1
        self.scores = (final_score - final_score.mean()) / final_score.std()
        self.scores = self.scores.dropna()
        
        return self.scores

    def calculate_momentum_scores(self) -> pd.Series:
        """Calculate momentum scores using the selected method."""
        if self.momentum_method == 'enhanced':
            return self.calculate_enhanced_momentum()
        else:
            return self.calculate_simple_momentum()

    def get_factor_exposures(self) -> pd.DataFrame:
        """
        Get the individual factor exposures for analysis.
        Returns a DataFrame with all factor scores for each stock.
        """
        if self.momentum_method != 'enhanced':
            return pd.DataFrame({'Simple_Momentum_Score': self.scores})
            
        returns = self.price_data.pct_change()
        
        # Calculate all factors
        short_term = returns.rolling(window=21).sum().iloc[-1]
        medium_term = returns.rolling(window=126).sum().iloc[-1]
        long_term = self.price_data.shift(self.gap_days).pct_change(self.lookback_days - self.gap_days).iloc[-1]
        volatility = returns.ewm(span=self.lookback_days).std().iloc[-1]
        high_low = self.price_data.rolling(window=14).max() - self.price_data.rolling(window=14).min()
        trend_strength = (high_low / self.price_data.rolling(window=14).mean()).iloc[-1]
        
        # Create factor exposure DataFrame
        factor_exposures = pd.DataFrame({
            'Short_Term_Momentum': short_term,
            'Medium_Term_Momentum': medium_term,
            'Long_Term_Momentum': long_term,
            'Volatility': volatility,
            'Trend_Strength': trend_strength,
            'Final_Score': self.scores
        })
        
        return factor_exposures

    def select_top_stocks(self) -> pd.Series:
        """Select top N stocks based on momentum scores."""
        if self.scores is None:
            self.calculate_momentum_scores()
        
        self.top_stocks = self.scores.sort_values(ascending=False).head(self.n_stocks)
        return self.top_stocks

    def simulate_portfolio(self) -> pd.Series:
        """Simulate equal-weighted portfolio returns."""
        if self.top_stocks is None:
            self.select_top_stocks()
            
        portfolio_prices = self.price_data[self.top_stocks.index]
        portfolio_returns = portfolio_prices.pct_change().dropna()
        equal_weights = np.repeat(1 / self.n_stocks, self.n_stocks)
        
        self.portfolio_value = (portfolio_returns @ equal_weights).add(1).cumprod()
        return self.portfolio_value

    def get_benchmark_returns(self) -> pd.Series:
        """Get SPY benchmark returns for comparison."""
        if self.portfolio_value is None:
            self.simulate_portfolio()
            
        spy = yf.download(
            "SPY", 
            start=self.portfolio_value.index[0], 
            end=self.portfolio_value.index[-1], 
            progress=False
        )['Close']
        
        self.spy_returns = (1 + spy.pct_change().dropna()).cumprod()
        return self.spy_returns

    def plot_results(self):
        """Plot portfolio performance against benchmark."""
        if self.portfolio_value is None:
            self.simulate_portfolio()
        if self.spy_returns is None:
            self.get_benchmark_returns()

        plt.figure(figsize=(12, 6))
        plt.plot(self.portfolio_value.index, self.portfolio_value, label="Momentum Portfolio")
        plt.plot(self.spy_returns.index, self.spy_returns, label="SPY")
        plt.title("Momentum Portfolio vs SPY")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def save_results(self, filename: str = None):
        """Save top stocks and their scores to CSV with timestamp."""
        if self.top_stocks is None:
            self.select_top_stocks()
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"momentum_portfolio_{timestamp}.csv"
        
        # Get factor exposures for top stocks
        factor_exposures = self.get_factor_exposures()
        top_stocks_exposures = factor_exposures.loc[self.top_stocks.index]
        
        # Save to CSV
        top_stocks_exposures.to_csv(filename)
        print(f"\nResults saved to {filename}")

    def compare_with_previous_month(self):
        """Compare current results with previous month's results."""
        if self.top_stocks is None:
            self.select_top_stocks()
            
        # Calculate previous month's date range
        end_date = datetime.today() - timedelta(days=30)
        start_date = end_date - timedelta(days=365 + 30)
        
        # Create a temporary screener for previous month
        prev_screener = MomentumScreener(
            n_stocks=self.n_stocks,
            lookback_days=self.lookback_days,
            gap_days=self.gap_days,
            momentum_method=self.momentum_method
        )
        
        # Load and process previous month's data
        tickers = self.load_tickers('cedears.txt')
        if not tickers:
            return
            
        print("\nCalculating previous month's results...")
        prev_screener.price_data = prev_screener.download_data(tickers, start_date, end_date)
        
        if prev_screener.price_data.empty:
            print("No previous month data available.")
            return
            
        # Calculate previous month's scores
        prev_screener.calculate_momentum_scores()
        prev_screener.select_top_stocks()
        
        # Get current and previous top stocks
        current_stocks = set(self.top_stocks.index)
        previous_stocks = set(prev_screener.top_stocks.index)
        
        # Calculate changes
        added_stocks = current_stocks - previous_stocks
        removed_stocks = previous_stocks - current_stocks
        
        # Print comparison results
        print("\n=== Portfolio Changes ===")
        print(f"Current Date: {datetime.today().strftime('%Y-%m-%d')}")
        print(f"Previous Date: {end_date.strftime('%Y-%m-%d')}")
        print("\nCurrent Top Stocks:")
        for ticker in self.top_stocks.index:
            change = "↑" if ticker in added_stocks else "↓" if ticker in removed_stocks else "→"
            print(f"{change} {ticker}: {self.top_stocks[ticker]:.2f}")
            
        print("\nChanges:")
        if added_stocks:
            print("\nAdded Stocks:")
            for ticker in added_stocks:
                print(f"+ {ticker}: {self.top_stocks[ticker]:.2f}")
        else:
            print("\nNo new stocks added")
            
        if removed_stocks:
            print("\nRemoved Stocks:")
            for ticker in removed_stocks:
                print(f"- {ticker}: {prev_screener.top_stocks[ticker]:.2f}")
        else:
            print("\nNo stocks removed")
            
        # Save comparison to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_filename = f"momentum_comparison_{timestamp}.csv"
        
        # Get common stocks between current and previous periods
        common_stocks = list(current_stocks.intersection(previous_stocks))
        
        comparison_data = pd.DataFrame({
            'Current_Score': self.top_stocks[common_stocks],
            'Previous_Score': prev_screener.top_stocks[common_stocks],
            'Status': ['Maintained' for _ in common_stocks]
        })
        
        # Add new and removed stocks separately
        if added_stocks:
            added_data = pd.DataFrame({
                'Current_Score': self.top_stocks[list(added_stocks)],
                'Previous_Score': np.nan,
                'Status': ['Added' for _ in added_stocks]
            })
            comparison_data = pd.concat([comparison_data, added_data])
            
        if removed_stocks:
            removed_data = pd.DataFrame({
                'Current_Score': np.nan,
                'Previous_Score': prev_screener.top_stocks[list(removed_stocks)],
                'Status': ['Removed' for _ in removed_stocks]
            })
            comparison_data = pd.concat([comparison_data, removed_data])
        
        comparison_data.to_csv(comparison_filename)
        print(f"\nComparison results saved to {comparison_filename}")

def main():
    # Initialize screener with simple momentum calculation
    screener = MomentumScreener(
        n_stocks=10,
        momentum_method='simple'  # Using simple momentum method
    )
    
    # Set date range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 + 30)
    
    # Load and process data
    tickers = screener.load_tickers('cedears.txt')
    if not tickers:
        return
        
    print("Downloading price data...")
    screener.price_data = screener.download_data(tickers, start_date, end_date)
    
    if screener.price_data.empty:
        print("No data available. Please check your internet connection and try again.")
        return
    
    # Calculate and display results
    screener.calculate_momentum_scores()
    screener.select_top_stocks()
    print(f"\nTop stocks by {screener.momentum_method} momentum score:")
    print(screener.top_stocks)
    
    # Compare with previous month
    screener.compare_with_previous_month()
    
    # Simulate and plot portfolio
    screener.simulate_portfolio()
    screener.get_benchmark_returns()
    screener.plot_results()
    
    # Save results
    screener.save_results()

if __name__ == "__main__":
    main()