import os
import json
import pandas as pd
import importlib

from datetime import datetime
from collected_item import CollectedItem
from data_collector.collector import Collector


class EuroCourtCasesCollector(Collector):
    ''''
    Using a kaggle dataset. These cases were written between 2001 and 2019
    https://www.kaggle.com/datasets/mathurinache/ecthrnaacl2021?select=dev.jsonl
    '''

    def __init__(self, root_path):
        super().__init__(root_path)

    def init(self):
        pass

    def load_jsonl(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)

    def collect(self, force=False):
        super().collect('EURO COURT CASES')
        
        input, output, meta = self.get_collection_paths()
        os.makedirs(output, exist_ok=True)
        os.makedirs(input, exist_ok=True)

        total_counter = 0
        
        # Let's check if we already collected this data. Then we dont need to again
        if not force and os.path.exists(meta):
            self.read_meta_file()
            print(f'{self.meta['total_collected']} cases were already collected at {self.meta['collected_at']}, hence we skip a redundant collection.')
            print("If you'd like to collect anyway, set the force variable to True.")
            return
        
        # The data is in jsonl format, so handle that explicit
        # I wouldn't use the dev.jsonl. It contains redundant cases I'd say.
        in_df = pd.concat([
            self.load_jsonl(os.path.join(input, 'train.jsonl')),
            self.load_jsonl(os.path.join(input, 'test.jsonl')),
        ])
        items = []
        print('Min Date:' + str(pd.to_datetime(in_df['judgment_date']).min()))
        print('Max Date:' + str(pd.to_datetime(in_df['judgment_date']).max()))

        for index, row in in_df.iterrows():
            facts = row['facts']
            item = CollectedItem(
                text='\n'.join(facts),
                chunks=facts,
                domain='euro_court_cases',
                date=row['judgment_date'],
                source='https://www.kaggle.com/datasets/mathurinache/ecthrnaacl2021/data',
                lang='en-EN'
            )
            items.append(item)
            total_counter += 1

            if total_counter % 300 == 0 and total_counter != 0:
                df = pd.DataFrame([item.__dict__ for item in items])
                df.to_json(os.path.join(output, f"cases_{total_counter // 300}.json"))
                print(f"Cases collected: {total_counter}/{len(in_df)}")
                items = []

        df = pd.DataFrame([item.__dict__ for item in items])
        total_counter += len(df)
        df.to_json(os.path.join(output, "cases_last.json"))
        print(f"Cases collected:{total_counter}/{total_counter}")

        # Save some metadata about this collection
        self.write_meta_file(total_counter, datetime.now())
        print(f'Collection finished. Total Cases collected: {total_counter}')

    def get_folder_path(self):
        return "euro_court_cases"
