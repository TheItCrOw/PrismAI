import os
import json
import pandas as pd

from collected_item import CollectedItem
from abc import ABC, abstractmethod

class Collector(ABC):
    
    def __init__(self, root_path):
        self.data_root_path = root_path
        self.meta = None
        self.collected_items_dfs = pd.DataFrame()
        self.synthetization_prompt = None

    def get_collection_paths(self):
        input = os.path.join(self.get_data_path(), 'raw')
        output = os.path.join(self.get_data_path(), 'collected')
        meta = os.path.join(self.get_data_path(), 'meta.json')
        return input, output, meta
    
    def get_synthesized_path(self):
        path = os.path.join(self.get_data_path(), 'synthesized')
        os.makedirs(path, exist_ok=True)
        return path

    def get_data_path(self):
        return os.path.join(self.data_root_path, self.get_folder_path())

    def read_meta_file(self):
        with open(os.path.join(self.get_data_path(), 'meta.json')) as handle:
            self.meta = json.loads(handle.read())
            return self.meta
        
    def write_meta_file(self, total_collected, date, total_synthesized=0):
        self.meta = {
            'collected_at': str(date),
            'total_collected': total_collected,
            'total_synthesized': total_synthesized
        }
        with open(os.path.join(self.get_data_path(), 'meta.json'), 'w') as fp:
            json.dump(self.meta, fp)
        self.collected = True

    def read_collected_from_cache(self, force=False):
        '''
        Reads the collected items, that are lying in the cache, into the RAM
        '''
        if (len(self.collected_items_dfs) > 0) and not force:
            print('Items already read from cache, skipping. If you want to re-read them by force, use force=True')
            return
        
        _, output, meta = self.get_collection_paths()
        dataframes = []

        for filename in os.listdir(output):
            if filename.endswith('.json'): 
                file_path = os.path.join(output, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    dataframes.append(pd.DataFrame(data))

        if dataframes:
            self.collected_items_dfs = dataframes
            #self.items_df = pd.concat(dataframes, ignore_index=True)
        else:
            self.collected_items_dfs = pd.DataFrame()

    def get_collected_items_dfs(self) -> pd.DataFrame:
        '''
        Reads the collected items from the cache and returns a list of dataframes.
        Each dataframe serves as a smaller chunk of the whole data as to reduce large files.
        '''
        if(len(self.collected_items_dfs) == 0):
            self.read_collected_from_cache()
        return self.collected_items_dfs

    def get_count(self):
        if self.meta is None:
            raise BaseException("The collector hasn't collected yet - can't get the count. Call 'collect()' first.")
        return int(self.meta['total_collected'])

    def get_synthetization_system_prompt(self, params):
        '''
        Reads a prompt file and fills in placeholders with values from the params dictionary.
        '''
        if self.synthetization_prompt is None:
            prompt_file = self.get_folder_path() + '.md'
            prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', prompt_file)
            
            if not os.path.exists(prompt_path):
                raise FileNotFoundError(f"Couldn't find prompt file under '{prompt_path}'.")
            
            with open(prompt_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            for key, value in params.items():
                placeholder = f"[{key.upper()}]"  # Match placeholders like [PLACEHOLDER]
                self.synthetization_prompt = content.replace(placeholder, str(value))

        return self.synthetization_prompt

    @abstractmethod
    def get_folder_path(self):
        pass

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def collect(self, name):
        print(f'\n\n ============== Collection started for {name} ============== \n')

