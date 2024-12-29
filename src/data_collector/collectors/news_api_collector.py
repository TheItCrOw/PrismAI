import os
import json
import pandas as pd
import importlib
import requests
import lxml.html

from newspaper import Article
from datetime import datetime
from collected_item import CollectedItem
from data_collector.collector import Collector


class NewsAPICollector(Collector):
    '''
    OBSOLETE COLLECTOR. We instead used a CNN dataset of news articles.
    '''

    query_url = 'https://newsapi.org/v2/everything?q={SEARCHQUERY}}&apiKey={APIKEY}&from={FROM}&to={TO}&page={PAGE}&pageSize={PAGESIZE}'

    def __init__(self, root_path, api_key):
        super().__init__(root_path)
        self.api_key = api_key

    def init(self):
        pass

    def collect(self, force=False):
        super().collect('NEWS API')
        
        input, output, meta = self.get_collection_paths()
        os.makedirs(output, exist_ok=True)
        os.makedirs(input, exist_ok=True)

        # we iterate through the exported speeches and get the their texts
        total_counter = 0
        
        # Let's check if we already collected this data. Then we dont need to again
        if not force and os.path.exists(meta):
            self.read_meta_file()
            print(f'{self.meta['total_collected']} news articles were already collected at {self.meta['collected_at']}, hence we skip a redundant collection.')
            print("If you'd like to collect anyway, set the force variable to True.")
            return
        
        counter = 1
        for i in range(1, 101):
            url = self.query_url.replace('{SEARCHQUERY}', 'bitcoin') \
                                .replace('{APIKEY}', self.api_key) \
                                .replace('{FROM}', '2017-01-01') \
                                .replace('{TO}', '2020-01-01') \
                                .replace('{PAGE}', str(i)) \
                                .replace('{PAGESIZE}', str(100)) \
                                
            response1 = requests.get(url)
            response1.raise_for_status() 
            articles = response1.json().get("articles", [])

            if articles:
                # Take the first search result
                first_result = articles[0]
                article_url = first_result.get("url")

                if article_url:
                    # Fetch the HTML content of the article
                    response2 = requests.get(article_url)
                    response2.raise_for_status()

                    # Use newspaper3k to parse the article content
                    article = Article(article_url)
                    article.download(input_html=response2.text)
                    article.parse()

                    # Print the article content
                    print(article.text)
                else:
                    print("No URL found in the first result.")
            else:
                print("No articles found in the API response.")

            return

        # Save some metadata about this collection
        self.write_meta_file(total_counter, datetime.now())
        
        print(f'Collection finished. Total News Articles collected: {total_counter}')

    def get_folder_path(self):
        return "news_api"
