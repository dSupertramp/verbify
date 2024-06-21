import unittest
from textgenie import advanced_sentiment_analysis


class TestSentimentAnalysis(unittest.TestCase):
    def test_advanced_sentiment_analysis(self):
        text = "The movie was not great, but I loved the acting."
        sentiment = advanced_sentiment_analysis(text)
        self.assertTrue("label" in sentiment[0] and "score" in sentiment[0])


if __name__ == "__main__":
    unittest.main()
