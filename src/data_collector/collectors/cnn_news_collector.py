import os
import json
import pandas as pd
import importlib

from datetime import datetime
from collected_item import CollectedItem
from data_collector.collector import Collector


class CNNNewsCollector(Collector):
    ''''
    Using a kaggle CNN news article dataset. The articles were written between 2007 and 2015
    https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail
    '''

    def __init__(self, root_path):
        super().__init__(root_path)

    def init(self):
        pass

    def collect(self, force=False):
        super().collect('CNN NEWS ARTICLES')
        
        input, output, meta = self.get_paths()
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
        
        items = []
        in_df = pd.read_csv(os.path.join(input, 'train.csv'))

        for index, row in in_df.iterrows():
            article = row['article']
            item = CollectedItem(
                text=article,
                chunks=article.split('\n'),
                domain='cnn_news',
                date='2007-2015',
                source='https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail',
                lang='en-EN'
            )
            items.append(item)
            total_counter += 1

            if total_counter % 1000 == 0 and total_counter != 0:
                df = pd.DataFrame([item.__dict__ for item in items])
                df.to_json(os.path.join(output, f"articles_{total_counter // 1000}.json"))
                print(f"CNN News articles collected: {total_counter}/{len(in_df)}")
                items = []

        df = pd.DataFrame([item.__dict__ for item in items])
        total_counter += len(df)
        df.to_json(os.path.join(output, "articles_last.json"))
        print(f"CNN News articles collected:{total_counter}/{total_counter}")

        # Save some metadata about this collection
        self.write_meta_file(total_counter, datetime.now())
        print(f'Collection finished. Total News Articles collected: {total_counter}')

    def get_folder_path(self):
        return "cnn_news"
