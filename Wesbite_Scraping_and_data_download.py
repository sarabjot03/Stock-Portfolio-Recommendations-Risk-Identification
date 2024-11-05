import yfinance as yf
import pandas as pd

# List of stock symbols (You can add more symbols here)
stocks = ["AAPL", "MSFT", "GOOGL"]

# Date range: 1 Jan 2015 to today's date
start_date = "2015-01-01"
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')  # Get today's date in YYYY-MM-DD format

for stock in stocks:
    ticker = yf.Ticker(stock)
    data = ticker.history(start=start_date, end=end_date)  # Download historical data
    data.to_csv(f"/Users/sarabjotsingh/Downloads/{stock}_data.csv")  # Save each stock's data to a CSV file
    print(f"Downloaded data for {stock} from {start_date} to {end_date}")

# Date range: 1 Jan 2015 to today's date
start_date = "1970-01-01"
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')  # Get today's date in YYYY-MM-DD format

for stock in stocks:
    ticker = yf.Ticker(stock)
    data = ticker.history(start=start_date, end=end_date)  # Download historical data
    data.to_csv(f"/Users/sarabjotsingh/Downloads/{stock}_data.csv")  # Save each stock's data to a CSV file
    print(f"Downloaded data for {stock} from {start_date} to {end_date}")

# Filter out any non-string values (like NaNs or None)
stocks_filtered = [stock for stock in stocks if isinstance(stock, str)]

# Loop over the cleaned list of stock symbols
for stock in stocks_filtered:
    try:
        ticker = yf.Ticker(stock)
        data = ticker.history(start=start_date, end=end_date)  # Download historical data
        data.to_csv(f"/Users/sarabjotsingh/Downloads/{stock}_data.csv")  # Save each stock's data to a CSV file
        print(f"Downloaded data for {stock} from {start_date} to {end_date}")
    except Exception as e:
        print(f"Failed to download data for {stock}: {e}")

