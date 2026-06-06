import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://127.0.0.1:8001")


class AnalysisPrompt:
    def __init__(self, current_price, sources, ticker) -> None:
        format_sources = ""
        for source in sources:
            format_sources += (
                f"Sentiment: {source.get('sentiment_label')} ({source.get('sentiment_score')})\n"
                f"Content: {source['content']} \n\n"
            )

        self.prompt = """
            You are a financial analyst specializing in stock market analysis. You are tasked with summarizing and analyzing a specific stock and providing a sentiment analysis based on given sources
            Sentiment_score_definition: x <= -0.35: Bearish; -0.35 < x <= -0.15: Somewhat-Bearish; -0.15 < x < 0.15: Neutral; 0.15 <= x < 0.35: Somewhat_Bullish; x >= 0.35: Bullish,
            stock symbol: {ticker}
            Current Price is {current_price}
            Relavant News: {sources}

            Note:Use plain text no bold or italics or any special characters

            **Example Format Response:**
            ---
            ## Stock Summary and Analysis:
            - Company Overview: XYZ is a leading provider of [products/services], operating primarily in [industry/sector]. Recently, the company has [recent significant events]. [source][url]
            - Financial Performance: The company reported a revenue of \$X million in the last quarter, with a profit margin of Y%. EPS was \$Z, showing an increase/decrease compared to the previous quarter. [source][url]
            - Market Trends: The [industry/sector] is currently experiencing [describe relevant trends]. [brief analysis]. Regulatory changes such as [regulatory factors] may impact the company. [source][url]
            - Analyst Ratings: Analysts have given the stock a rating of [rating], with a price target range of \$[low] to \$[high]. [source][url]
            - Sentiment: Based on the current news, We can say that it is [bearish or bullish]
            Sentiment Analysis:

            - Positive Sentiments: [Summary of positive sentiments]
            - Negative Sentiments: [Summary of negative sentiments]
            - Neutral Sentiments: [Summary of neutral sentiments]

            Conclusion:

            - Overall Sentiment: The overall sentiment towards XYZ is [positive/negative/neutral].
            - Investment Recommendation: Based on the analysis, it is recommended to [buy/hold/sell] the stock. This recommendation is supported by [key points from the analysis].

            > Note: The above analysis is based on publicly available information and should not be taken as personalized investment advice.
        """.format(
            ticker=ticker,
            sources=format_sources,
            current_price=current_price,
        )

    def __str__(self) -> str:
        return self.prompt


class AnalysisAgent:
    def __init__(self) -> None:
        pass

    def get_analysis(self, ticker, sources, current_price):
        prompt = AnalysisPrompt(
            ticker=ticker, current_price=current_price, sources=sources
        )

        try:
            res = requests.post(
                LLM_SERVICE_URL + "/generate",
                json={"content": prompt.prompt},
                timeout=300,
            )
            res.raise_for_status()
            analysis = res.json()
            return analysis["content"]

        except requests.exceptions.RequestException as e:
            detail = res.text if "res" in locals() and res is not None else str(e)
            return f"[Error: LLM generation failed: {detail}]"

    def run(self, data: dict):
        analysis = self.get_analysis(
            ticker=data["ticker"], sources=data["sources"], current_price=data["price"]
        )
        data["analysis"] = analysis
        return data
