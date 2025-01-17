import os
import json
import pandas as pd
import importlib
import nltk

from datetime import datetime
from data_collector.collected_item import CollectedItem
from data_collector.collector import Collector


class BlogCorpusCollector(Collector):
    ''''
    Using a kaggle article dataset that needs to be cited! The blogs were written in 2004
    https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus?select=blogtext.csv
    '''

    def __init__(self, root_path):
        super().__init__(root_path)

    def init(self):
        nltk.download('punkt')
        nltk.download('punkt_tab')

    def split_into_sentences(self, text):
        sentences = nltk.sent_tokenize(text)
        return sentences

    def collect(self, force=False):
        super().collect('BLOG AUTHORSHIP CORPUS')
        
        input, output, meta = self.get_collection_paths()
        os.makedirs(output, exist_ok=True)
        os.makedirs(input, exist_ok=True)

        # we iterate through the exported speeches and get the their texts
        total_counter = 0
        
        # Let's check if we already collected this data. Then we dont need to again
        if not force and os.path.exists(meta):
            self.read_meta_file()
            print(f'{self.meta['total_collected']} blogs were already collected at {self.meta['collected_at']}, hence we skip a redundant collection.')
            print("If you'd like to collect anyway, set the force variable to True.")
            return
        
        items = []
        max_blogs = 200000
        in_df = pd.read_csv(os.path.join(input, 'blogtext.csv'))

        for index, row in in_df.iterrows():
            blog = row['text']
            item = CollectedItem(
                text=blog,
                chunks=self.split_into_sentences(blog),
                domain='blog_authorship_corpus',
                date=row['date'],
                source='https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus?select=blogtext.csv',
                lang='en-EN'
            )
            items.append(item)
            total_counter += 1

            if total_counter % 10000 == 0 and total_counter != 0:
                df = pd.DataFrame([item.__dict__ for item in items])
                df.to_json(os.path.join(output, f"blogs_{total_counter // 10000}.json"))
                print(f"Blog posts collected: {total_counter}/{len(in_df)}")
                items = []
            
            if(total_counter > max_blogs):
                break

        df = pd.DataFrame([item.__dict__ for item in items])
        if len(df) > 0:
            total_counter += len(df)
            df.to_json(os.path.join(output, "blogs_last.json"))
            print(f"Blog posts collected:{total_counter}/{total_counter}")

        # Save some metadata about this collection
        self.write_meta_file(total_counter, datetime.now())
        print(f'Collection finished. Total Blogs collected: {total_counter}')

    def get_folder_path(self):
        return "blog_authorship_corpus"
