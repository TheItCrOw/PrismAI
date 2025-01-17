import os
import json
import pandas as pd
import importlib
import bz2
import traceback
import warnings

from cassis import load_cas_from_xmi, load_typesystem, load_dkpro_core_typesystem, merge_typesystems
from datetime import datetime
from data_collector.collected_item import CollectedItem
from data_collector.collector import Collector

warnings.filterwarnings(
    "ignore", 
    message="Not mapping external offset", 
    module="cassis.cas"
)

class GutenbergCollector(Collector):

    def __init__(self, root_path):
        super().__init__(root_path)

    def init(self):
        pass

    def process_xmi_content(self, xmi_content, typesystem):
        '''
        See: https://github.com/dkpro/dkpro-cassis
        '''

        cas = load_cas_from_xmi(xmi_content, typesystem=typesystem)
        chunks = []
        for paragraph in cas.select('de.tudarmstadt.ukp.dkpro.core.api.structure.type.Paragraph'):
            chunks.append(paragraph.get_covered_text())

        title = cas.select('de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData')[0]['documentTitle']
        return title, chunks

    def collect(self, force=False):
        super().collect('GUTENBERG')
        
        _, output, meta = self.get_collection_paths()
        # This collector is different, as we already have the Gutenberg files in
        # XMI on our local storage, so THAT is our input directory
        inputs = [
            {'path':'/storage/xmi/Gutenberg/html/single/de', 'lang': 'de-DE'}, 
            {'path':'/storage/xmi/Gutenberg/html/single/en', 'lang': 'en-EN'}, 
        ]
        os.makedirs(output, exist_ok=True)

        # we iterate through the exported speeches and get the their texts
        total_counter = 0
        max_books = 35000
        items = []
        
        # Let's check if we already collected this data. Then we dont need to again
        if not force and os.path.exists(meta):
            self.read_meta_file()
            print(f'{self.meta['total_collected']} books were already collected at {self.meta['collected_at']}, hence we skip a redundant collection.')
            print("If you'd like to collect anyway, set the force variable to True.")
            return
        
        # We need to fetch the TypSystem first (This is UIMA stuff)
        with open(os.path.join(self.get_data_path(), 'TypeSystem.xml'), 'rb') as f:
            typesystem = load_typesystem(f)

        print('Reading in with the following typesystems:')
        for t in typesystem.get_types():
            print(t)

        for input in inputs:
            for root, _, files in os.walk(input['path']):
                if total_counter >= max_books:
                    break

                for file in files:
                    if file.endswith('.bz2'):
                        file_path = os.path.join(root, file)

                        # Decompress the bz2 compression
                        try:
                            with bz2.open(file_path, 'rt') as bz2_file:
                                xmi_content = bz2_file.read()
                                # Yeah, for some reason this is totally fked and cost me 2h to figure out.
                                # it should be XMI:ID not "just" id as the XMI files have them! So manually
                                # replace those occurences ._.
                                xmi_content = xmi_content.replace('<cas:NULL id="0"/>', '<cas:NULL xmi:id="0"/>')
                                title, chunks = self.process_xmi_content(xmi_content, typesystem)

                                item = CollectedItem(
                                    text=title + '\n' + '\n'.join(chunks),
                                    chunks=chunks,
                                    domain='gutenberg',
                                    date='-',
                                    source=file_path,
                                    lang=input['lang']
                                )
                                items.append(item)
                                total_counter += 1

                                if total_counter % 50 == 0 and total_counter != 0:
                                    df = pd.DataFrame([item.__dict__ for item in items])
                                    df.to_json(os.path.join(output, f"books_{total_counter // 100}.json"))
                                    print(f"Books collected: {total_counter}/{max_books}")
                                    items = []

                                if total_counter >= max_books:
                                    break

                        except Exception as ex:
                            print("Couldn't open/extract the XMI file.")
                            print(ex)
                            print(traceback.format_exc())
                            print('Total books collected: ' + str(total_counter))

        df = pd.DataFrame([item.__dict__ for item in items])
        total_counter += len(df)
        df.to_json(os.path.join(output, "books_last.json"))
        print(f"Books collected:{total_counter}/{max_books}")

        # Save some metadata about this collection
        self.write_meta_file(total_counter, datetime.now())
        
        print(f'Collection finished. Total Books collected: {total_counter}')

    def get_folder_path(self):
        return "gutenberg"
