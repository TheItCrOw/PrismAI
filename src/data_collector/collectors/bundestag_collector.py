import os
import json
import pandas as pd
import importlib

from datetime import datetime
from collected_item import CollectedItem
from data_collector.collector import Collector

class BundestagCollector(Collector):

    def __init__(self, root_path):
        super().__init__(root_path)

    def init(self):
        pass

    def collect(self, force=False):
        super().collect('BUNDESTAG')
        
        input, output, meta = self.get_collection_paths()
        os.makedirs(output, exist_ok=True)

        # we iterate through the exported speeches and get the their texts
        total_files = len([name for name in os.listdir(input)])
        total_counter = 0
        
        # Let's check if we already collected this data. Then we dont need to again
        if not force and os.path.exists(meta):
            self.read_meta_file()
            print(f'{self.meta['total_collected']} speeches were already collected at {self.meta['collected_at']}, hence we skip a redundant collection.')
            print("If you'd like to collect anyway, set the force variable to True.")
            return
        
        counter = 1
        for name in os.listdir(input):
            items = []
            file_path = os.path.join(input, name)
            with open(file_path) as handle:
                protocol = json.loads(handle.read())
            speeches = protocol['NLPSpeeches']

            for speech in speeches:
                try:
                    item = CollectedItem(
                        text=speech['Text'],
                        chunks=list(map(lambda s: s['Text'], speech['Segments'])),
                        domain='bundestag',
                        date=protocol['Protocol']['Date'],
                        source='https://bundestag-mine.de/api/DashboardController/GetNLPSpeechById/' + speech['Id'],
                        lang='de-DE'
                    )
                    items.append(item)
                except Exception as ex:
                    print("Couldn't collect speech with Text " + speech['Text'][:30] + "...")
                    print(ex)
        
            df = pd.DataFrame([item.__dict__ for item in items])
            total_counter += len(df)
            df.to_json(os.path.join(output, str(counter) + ".json"))
            print(f"Protocols collected: {counter}/{total_files}")
            counter += 1

        # Save some metadata about this collection
        self.write_meta_file(total_counter, datetime.now())
        
        print(f'Collection finished. Total Speeches collected: {total_counter}')

    def get_folder_path(self):
        return "bundestag"