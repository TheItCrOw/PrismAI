import pandas as pd

from collector import Collector
from collected_item import CollectedItem

class CollectionVisualizer:
    '''
    Implements some standard visualization methods for any kind of collector
    '''

    def __init__(self, items_df : pd.DataFrame):
        self.items_df = items_df