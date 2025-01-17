import os
import json
import pandas as pd

from data_collector.collected_item import CollectedItem
from abc import ABC, abstractmethod

class Collector(ABC):
    
    def __init__(self, root_path):
        self.data_root_path = root_path
        self.meta = self.read_meta_file()
        self.collected_items_dfs = pd.DataFrame()
        self.synthesized_items_dfs = pd.DataFrame()
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
    
    def get_feature_space_path(self):
        path = os.path.join(self.get_data_path(), 'feature_space')
        os.makedirs(path, exist_ok=True)
        return path

    def get_data_path(self):
        return os.path.join(self.data_root_path, self.get_folder_path())

    def read_meta_file(self):
        path = os.path.join(self.get_data_path(), 'meta.json')
        if not os.path.exists(path): 
            return None

        with open(path) as handle:
            self.meta = json.loads(handle.read())
            return self.meta

    def get_meta_file(self):
        return self.meta

    def update_meta_file(self, meta):
        with open(os.path.join(self.get_data_path(), 'meta.json'), 'w') as fp:
            json.dump(meta, fp)

    def write_meta_file(self, total_collected, date, total_synthesized=0, total_feature_extracted=0):
        self.meta = {
            'collected_at': str(date),
            'total_collected': total_collected,
            'total_synthesized': total_synthesized,
            'total_feature_extracted': total_feature_extracted
        }
        with open(os.path.join(self.get_data_path(), 'meta.json'), 'w') as fp:
            json.dump(self.meta, fp)
        self.collected = True

    def read_from_cache(self, level):
        '''
        Reads items of various levels (raw, collected, synthesized), that are lying in the cache, into the RAM
        '''
        path = os.path.join(self.get_data_path(), level)
        if not os.path.exists(path):
            return pd.DataFrame
        
        dataframes = []

        for filename in os.listdir(path):
            if filename.endswith('.json'): 
                file_path = os.path.join(path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    dataframes.append(pd.DataFrame(data))

        if dataframes:
            return dataframes
            #return pd.concat(dataframes, ignore_index=True)
        else:
            return pd.DataFrame()

    def get_synthesized_items_dfs(self) -> pd.DataFrame:
        '''
        Reads the synthesized items from cache and returns a list of dataframes.
        '''
        if(len(self.synthesized_items_dfs) == 0):
            self.synthesized_items_dfs = self.read_from_cache('synthesized')
        return self.synthesized_items_dfs

    def get_collected_items_dfs(self) -> pd.DataFrame:
        '''
        Reads the collected items from the cache and returns a list of dataframes.
        Each dataframe serves as a smaller chunk of the whole data as to reduce large files.
        '''
        if(len(self.collected_items_dfs) == 0):
            self.collected_items_dfs = self.read_from_cache('collected')
        return self.collected_items_dfs

    def get_synthesized_count(self):
        if self.synthesized_items_dfs is None:
            raise BaseException("The collector hasn't synthesized yet - can't get the count. Call 'collect()' first.")
        return sum(len(df) for df in self.synthesized_items_dfs)

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

