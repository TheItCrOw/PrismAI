import pandas as pd

from data_collector.collector import Collector
from data_collector.collected_item import CollectedItem

class CollectionVisualizer:
    '''
    Implements some standard visualization methods for any kind of collector
    '''

    def __init__(self, items_df : pd.DataFrame):
        self.items_df = items_df