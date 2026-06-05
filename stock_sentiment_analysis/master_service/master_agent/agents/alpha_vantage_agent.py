import os
import requests
from dotenv import load_dotenv

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


class AlphaVantageNewsAgent:
    def __init__(self):
        self.api_key = ALPHA_VANTAGE_API_KEY

    def search(self, ticker):
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.api_key}"
        response = requests.get(url)
        if response.status_code != 200:
            print("❌ Alpha Vantage API error:", response.text)
            return []

        data = response.json()
        return data.get("feed", [])

    def run(self, data: dict):
        raw_articles = self.search(data["ticker"])

        ticker = data["ticker"]
        relevant_articles = []
        for article in raw_articles:
            if any(
                t["ticker"].upper() == ticker.upper()
                for t in article.get("ticker_sentiment", [])
            ):
                relevant_articles.append(
                    {
                        "title": article.get("title"),
                        "url": article.get("url"),
                        "content": article.get("summary", ""),
                        "sentiment_score": next(
                            (
                                t["ticker_sentiment_score"]
                                for t in article["ticker_sentiment"]
                                if t["ticker"].upper() == ticker.upper()
                            ),
                            None,
                        ),
                        "sentiment_label": next(
                            (
                                t["ticker_sentiment_label"]
                                for t in article["ticker_sentiment"]
                                if t["ticker"].upper() == ticker.upper()
                            ),
                            None,
                        ),
                    }
                )

        data["sources"] = relevant_articles
        return data
