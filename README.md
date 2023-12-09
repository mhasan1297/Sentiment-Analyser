# Financial News Summarisation and Sentiment Analysis

## Introduction
This project focuses on summarizing financial news articles and analyzing sentiments related to monitored stock tickers. It leverages the Pegasus model for text summarization and performs sentiment analysis on news articles related to specific stocks.

## Dependencies
Ensure that you have the following dependencies installed:
- `transformers`
- `bs4` (BeautifulSoup)
- `requests`

Install dependencies using the following command:
```bash
pip install transformers beautifulsoup4 requests
Setup Summarization Model
python
Copy code
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests

model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)
Note: Some weights of PegasusForConditionalGeneration were not initialized from the model checkpoint at human-centered-summarization/financial-summarization-pegasus. It is recommended to train this model on a down-stream task for better predictions.

Summarize Articles
python
Copy code
# Code sections 4.3 and 5

def summarize(articles):
    # Function definition here

summaries = {ticker: summarize(articles[ticker]) for ticker in monitored_tickers}
print(summaries)
Sentiment Analysis
python
Copy code
# Code section 5

from transformers import pipeline
sentiment = pipeline('sentiment-analysis')
scores = {ticker: sentiment(summaries[ticker]) for ticker in monitored_tickers}
print(scores)
Exporting Results to CSV
python
Copy code
# Code section 6

def create_output_array(summaries, scores, urls):
    # Function definition here

final_output = create_output_array(summaries, scores, cleaned_urls)
final_output.insert(0, ['Ticker', 'Summary', 'Label', 'Confidence', 'URL'])

import csv
with open('assetsummaries.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)
Note: The code provided is a skeleton, and you need to implement the function definitions for summarize and create_output_array. Additionally, adapt the project structure based on your specific needs and preferences.
