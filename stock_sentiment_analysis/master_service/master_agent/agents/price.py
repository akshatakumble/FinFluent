import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
API_KEY = os.getenv("STOCK_API_KEY")
URL = "https://api.twelvedata.com/price"

class PriceAgent:
    def __init__(self, url=URL, apikey=API_KEY) -> None:
        self.url = url
        self.api_key = apikey
    
    def get_price(self, ticker):
        API_KEY = self.api_key

        querystring = {"symbol": ticker, "apikey": API_KEY}

        response = requests.request("GET", self.url, params=querystring)
        response = response.json()
        if 'price' in response:
            return float(response['price'])
        else:
            return -1
    
    def run(self, data: dict):
        res = self.get_price(data["ticker"])
        data["price"] = res
        return data
        