
import os
import json
import requests
import pandas as pd
import regex as re
import importlib
import time
import lxml.html
import spiegel_scraper as spon

from datetime import datetime, date, timedelta
from data_collector.collected_item import CollectedItem
from data_collector.collector import Collector

class SpiegelAPI():

    def scrape_article_html(self, article_url):
        '''
        From an article url, builds an article object. This code was taken from:
        https://github.com/ietz/spiegel-scraper
        and had to be copy pasted since the repo had an error that returned nothing.
        '''

        article_html = requests.get(article_url).text
        doc = lxml.html.fromstring(article_html)

        ld_content = doc.xpath('string(//script[@type="application/ld+json"]/text())')
        ld = json.loads(ld_content)
        ld_by_type = {ld_entry['@type']: ld_entry for ld_entry in ld}
        news_ld = ld_by_type['NewsArticle']

        settings = json.loads(doc.xpath('string(//script[@type="application/settings+json"]/text())'))
        info = settings['editorial']['info']

        text_node_selector = \
            'main .word-wrap > p,'  \
            'main .word-wrap > h3, ' \
            'main .word-wrap > ul > li, ' \
            'main .word-wrap > ol > li'
        text_nodes = doc.cssselect(text_node_selector)
        text = re.sub(r'\n+', '\n', '\n'.join([node.text_content() for node in text_nodes])).strip()

        return {
            'url': doc.xpath('string(//link[@rel="canonical"]/@href)'),
            'id': info['article_id'],
            'channel': info['channel'],
            'subchannel': info['subchannel'],
            'headline': {
                'main': info['headline'],
                'social': info['headline_social']
            },
            'intro': info['intro'],
            'text': text,
            'topics': info['topics'],
            'author': settings['editorial']['author'],
            'date_created': news_ld['dateCreated'],
            'date_modified': news_ld['dateModified'],
            'date_published': news_ld['datePublished'],
            'breadcrumbs': [breadcrumb['item']['name'] for breadcrumb in ld_by_type['BreadcrumbList']['itemListElement']],
        }

    def get_articles_of_date(self, date):
        archive_entries = spon.archive.by_date(date)
        all_archives = []
        for entry in archive_entries:
            article_url = entry['url']
            try:
                article = self.scrape_article_html(article_url)
                all_archives.append(article)
            except Exception as ex:
                print("Skipped one article since an error occured")
                print(ex)
        return all_archives

class SpiegelCollector(Collector):

    def __init__(self, root_path):
        super().__init__(root_path)

    def init(self):
        pass

    def scrape(self, date, output_path):
        api = SpiegelAPI()
        articles = []
        items = []

        try:
            articles = api.get_articles_of_date(date)
        except Exception as ex:
            print("Error while fetching spiegel articles:")
            print(ex)
            print("Waiting a few seconds, this is probably due to timeout.")
            time.sleep(60)
            self.scrape(date)

        for article in articles:
            try:
                item = CollectedItem(
                    text=article['text'],
                    chunks=article['text'].split('\n'),
                    domain='spiegel_articles',
                    date=date,
                    source=article['url'],
                    lang='de-DE'
                )
                items.append(item)
            except Exception as ex:
                print("Couldn't collect an article, error: ")
                print(ex)

        df = pd.DataFrame([item.__dict__ for item in items])
        df.to_json(os.path.join(output_path, str(date) + ".json"))
        return len(df)

    def collect(self, force=False):
        super().collect('SPIEGEL ARTICLES')

        input, output, meta = self.get_collection_paths()
        os.makedirs(output, exist_ok=True)
        total_counter = 0
        
        # Let's check if we already collected this data. Then we dont need to again
        if not force and os.path.exists(meta):
            self.read_meta_file()
            print(f'{self.meta['total_collected']} articles were already collected at {self.meta['collected_at']}, hence we skip a redundant collection.')
            print("If you'd like to collect anyway, set the force variable to True.")
            return
        
        start_date = date(2020, 1, 1)
        end_date = date(2018, 1, 1)
        delta = timedelta(days=1)
        while start_date >= end_date:
            try:
                total_counter += self.scrape(start_date, output)
                start_date -= delta
                print('Stored articles for ' + str(start_date))
            except Exception as ex:
                print("Unknown error. Restarting in a bit.")
                time.sleep(60)

        # Save some metadata about this collection
        self.write_meta_file(total_counter, datetime.now())
        
        print(f'Collection finished. Total Articles collected: {total_counter}')

    def get_folder_path(self):
        return "spiegel_articles"
