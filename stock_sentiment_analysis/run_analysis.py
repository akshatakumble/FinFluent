import requests
import argparse

# Parse command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("--ticker", type=str, help="Stock ticker symbol (e.g. AAPL)")
args = parser.parse_args()

# Get ticker
ticker = args.ticker.strip().upper() if args.ticker else input("Enter a stock ticker (e.g., TSLA, AAPL, NVDA): ").strip().upper()

url = f"http://localhost:8000/ticker?ticker={ticker}"

# Send GET request
response = requests.get(url)

# Handle response
if response.ok:
    data = response.json()
    price = data.get("price", "N/A")
    print(f"\nStock Price: {price}")
    if isinstance(price, (int, float)) and price < 0:
        print(
            "Note: Live price unavailable. Add STOCK_API_KEY (Twelve Data) to "
            "stock_sentiment_analysis/master_service/.env"
        )
    print(f"\nAnalysis:\n{data.get('analysis', 'No analysis returned')}\n")
else:
    print("\nError:", response.status_code)
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    print(detail)
