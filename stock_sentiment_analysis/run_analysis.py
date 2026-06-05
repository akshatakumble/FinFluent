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
    print(f"\n📈 Stock Price: {data.get('price', 'N/A')}")
    print(f"\n🧠 Analysis:\n{data.get('analysis', 'No analysis returned')}\n")
else:
    print("\n❌ Error:", response.status_code)
    print(response.text)
