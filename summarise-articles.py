#### 1. Install and Import Baseline Dependencies

from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests

#### 2. Setup Summarisation Model (2.2GB Memory)

model_name = "human-centered-summarization/financial-summarization-pegasus" # Summarisation Model Downloaded 
tokenizer = PegasusTokenizer.from_pretrained(model_name) # De Coder
model = PegasusForConditionalGeneration.from_pretrained(model_name) # Summarise code ID

#### 3. Summarise a Single Article

url = "https://au.finance.yahoo.com/news/china-restricting-tesla-use-uncovers-a-significant-challenge-for-elon-musk-expert-161921664.html"
headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
r = requests.get(url, headers=headers)
soup = BeautifulSoup(r.text, 'html.parser')
paragraphs = soup.find_all('p') # Find all paragraphs

text = [paragraph.text for paragraph in paragraphs] # Strip the whole article to text format
words = ' '.join(text).split(' ')[:400] #First 400 words, current limit on summarisation model.
ARTICLE = ' '.join(words) # Join the 400 words back to make a shorter article

input_ids = tokenizer.encode(ARTICLE, return_tensors='pt') # En coding aricle to tensorflow ID's
output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True) # Generate the summary
summary = tokenizer.decode(output[0], skip_special_tokens=True) # De code the summary back to text format

#### 4. Building a News and Sentiment Pipeline

def get_user_tickers():
    while True:
        user_input = input("Enter stock or cryptocurrency symbols (comma-separated) or type 'done' to finish: ").strip().upper()

        if user_input == 'DONE':
            break

        tickers = [ticker.strip() for ticker in user_input.split(',')]
        
        if not tickers:
            print("Please enter at least one stock or cryptocurrency symbol.")
        else:
            return tickers

# Example usage:
monitored_tickers = get_user_tickers()
print("Monitored Tickers:", monitored_tickers)



#### 4.1. Search for Stock News using Google and Yahoo Finance

import re

def search_for_stock_news_urls(ticker):
    search_url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
    r = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    atags = soup.find_all('a')
    hrefs = [link['href'] for link in atags]

    # Strip out unwanted URLs
    pattern = re.compile(r'(/news/|/video/|/m/).*\.html')

    # Extract and print the matched URLs with the desired prefix
    filtered_urls = [f'https://finance.yahoo.com{url}' for url in hrefs if pattern.match(url)]
    return filtered_urls

#### 4.2. Search and Scrape Cleaned URLs

cleaned_urls = {ticker: search_for_stock_news_urls(ticker) for ticker in monitored_tickers}

def scrape_and_process(URLs):
    ARTICLES = []
    for url in URLs: 
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = [paragraph.text for paragraph in paragraphs]
        words = ' '.join(text).split(' ')[:350]
        ARTICLE = ' '.join(words)
        ARTICLES.append(ARTICLE)
    return ARTICLES

articles = {ticker:scrape_and_process(cleaned_urls[ticker]) for ticker in monitored_tickers}

#### 4.3. Summarise all Articles

def summarize(articles):
    summaries = []
    for article in articles:
        input_ids = tokenizer.encode(article, return_tensors='pt')
        output = model.generate(input_ids, max_length=55, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

summaries = {ticker:summarize(articles[ticker]) for ticker in monitored_tickers}

#### 5. Adding Sentiment Analysis

from transformers import pipeline
sentiment = pipeline('sentiment-analysis')

scores = {ticker:sentiment(summaries[ticker]) for ticker in monitored_tickers}

#### 6. Exporting Results to CSV

def create_output_array(summaries, scores, urls):
    output = []
    for ticker in monitored_tickers:
        for counter in range(len(summaries[ticker])):
            output_this = [
                ticker,
                summaries[ticker][counter],
                scores[ticker][counter]['label'],
                scores[ticker][counter]['score'],
                urls[ticker][counter]

            ]
            output.append(output_this)
    return output

final_output = create_output_array(summaries, scores, cleaned_urls)
final_output.insert(0, ['Ticker', 'Summary', 'Label', 'Confidence', 'URL'])

import csv
with open('assetsummaries.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting = csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)

