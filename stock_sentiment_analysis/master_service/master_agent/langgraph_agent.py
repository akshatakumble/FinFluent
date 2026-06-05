from master_agent.agents.alpha_vantage_agent import AlphaVantageNewsAgent  
from .agents import PriceAgent, AnalysisAgent



class MasterAgent:
    def __init__(self, ticker):
        self.data = {"ticker": ticker}
        self.agents = [
            PriceAgent(),  # still using Twelve Data API
            AlphaVantageNewsAgent(),  # now using Alpha Vantage for sentiment
            AnalysisAgent(),  # your LLM interaction agent
        ]

    def run(self):
        for agent in self.agents:
            self.data = agent.run(self.data)
        return self.data
