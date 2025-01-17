
import os
import json
import requests
import pandas as pd
import regex as re
import importlib
import time

from datetime import datetime, date, timedelta
from data_collector.collected_item import CollectedItem
from data_collector.collector import Collector

class OpenLegalDataCollector(Collector):

    query_url = 'https://de.openlegaldata.io/api/cases/?court__slug=&court__jurisdiction=&court__state=&court=&date_after=2015-01-01&o=date&court__level_of_appeal=&date_before=2020-01-01&format=json&page={PAGE}'

    def __init__(self, root_path):
        super().__init__(root_path)

    def init(self):
        pass

    def collect(self, force=False):
        super().collect('OPEN LEGAL DATA ARTICLES')

        input, output, meta = self.get_collection_paths()
        os.makedirs(output, exist_ok=True)
        os.makedirs(input, exist_ok=True)
        total_counter = 0
        items = []
        
        # Let's check if we already collected this data. Then we dont need to again
        if not force and os.path.exists(meta):
            self.read_meta_file()
            print(f'{self.meta['total_collected']} legal articles were already collected at {self.meta['collected_at']}, hence we skip a redundant collection.')
            print("If you'd like to collect anyway, set the force variable to True.")
            return
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }
        pages = 4000
        ex_catched = 0
        for i in range(1, pages + 1):
            try:
                url = self.query_url.replace('{PAGE}', str(i))
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

                for article in data['results']:
                    text = article['content']
                    item = CollectedItem(
                        text=text,
                        chunks=text.split('\n'),
                        domain='open_legal_data',
                        date=article['date'],
                        source=url,
                        lang='de-DE'
                    )
                    items.append(item)
                    total_counter += 1

                    if total_counter % 100 == 0 and total_counter != 0:
                        df = pd.DataFrame([item.__dict__ for item in items])
                        df.to_json(os.path.join(output, f"articles_{total_counter // 100}.json"))
                        print(f"Legal articles collected: {i}/{pages}")
                        items = []
        
                time.sleep(3)
            except Exception as ex:
                print('Exception thrown while fetching legal endpoints, probably request timeout.')
                print(ex)
                ex_catched += 1
            
            if(ex_catched > 10):
                print("Too many exceptions caught, we stop this collection and store the current data.")
                break

        df = pd.DataFrame([item.__dict__ for item in items])
        total_counter += len(df)
        df.to_json(os.path.join(output, "articles_last.json"))
        print(f"Legal articles collected:{total_counter}/{total_counter}")

        # Save some metadata about this collection
        self.write_meta_file(total_counter, datetime.now())
        print(f'Collection finished. Total Legal Articles collected: {total_counter}')

    def get_folder_path(self):
        return "open_legal_data"
