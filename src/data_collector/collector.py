import os
import json

from abc import ABC, abstractmethod

class Collector(ABC):
    
    data_root_path = ''

    def __init__(self, root_path):
        self.data_root_path = root_path
        self.meta = None
        pass

    def get_paths(self):
        input = os.path.join(self.get_data_path(), 'raw')
        output = os.path.join(self.get_data_path(), 'collected')
        meta = os.path.join(self.get_data_path(), 'meta.json')
        return input, output, meta

    def get_data_path(self):
        return os.path.join(self.data_root_path, self.get_folder_path())

    def read_meta_file(self):
        with open(os.path.join(self.get_data_path(), 'meta.json')) as handle:
            self.meta = json.loads(handle.read())
            return self.meta
        
    def write_meta_file(self, total_collected, date):
        self.meta = {
            'collected_at': str(date),
            'total_collected': total_collected
        }
        with open(os.path.join(self.get_data_path(), 'meta.json'), 'w') as fp:
            json.dump(self.meta, fp)

    @abstractmethod
    def get_folder_path(self):
        pass

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def collect(self, name):
        print(f'\n\n ============== Collection started for {name} ============== \n')

