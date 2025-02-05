import os
import sys
import pathlib
import json
import pandas as pd
import uuid
import re
import requests
import io
import concurrent.futures

from dotenv import load_dotenv
from tqdm.notebook import tqdm
from cassis import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mongodb_client import MongoDBConnection

load_dotenv()

class UIMAExporter():
    ''''
    An exporter object, that exports the collected items into a UIMA Cas XMI file.
    '''

    def __init__(self, root_data_path):
        self.root_data_path = root_data_path
        self.counters = {}
        self.__load_typesystems()

    def __sanitize_string(self, value):
        if isinstance(value, str):
            # Remove invalid XML characters
            value = re.sub(r'[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD]', '', value)
        return value

    def __load_typesystems(self):
        # This looks kinda ugly, but from the documentation, I don't know wny other way to load
        # the typesystems...
        with open('DocumentMetadata.xml', 'rb') as f:
            metadata_ts = load_typesystem(f)

        with open('TextTechnologyDokumentAnnotation.xml', 'rb') as f:
            doc_annotation_ts = load_typesystem(f)
        
        with open('UceDynamicMetadata.xml', 'rb') as f:
            uce_ts = load_typesystem(f)

        self.typesystem = merge_typesystems(metadata_ts, doc_annotation_ts, uce_ts)
        print('======= Loaded UIMA Typesystems:')
        for t in self.typesystem.get_types():
            print(t)

    def __build_xmi_from_collected_item(self, item):
        try:
            file_name = f'{item['domain']}_{item['id']}'
            cas = Cas(
                sofa_string = self.__sanitize_string(item['text']),
                document_language = item['lang'],
                typesystem = self.typesystem
            )
            UceDynamicMetadata = self.typesystem.get_type('org.texttechnologylab.annotation.uce.Metadata')
            DocumentAnnotation = self.typesystem.get_type('org.texttechnologylab.annotation.DocumentAnnotation')
            DocumentMetadata = self.typesystem.get_type('de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData')

            cas.add(DocumentMetadata(
                documentTitle=file_name, 
                documentId=item['id'], 
                documentUri='collected_items__' + item['source'] + item['id']
            ))
            cas.add(DocumentAnnotation(
                author='Bönisch, Kevin and Stoeckel, Manuel and Mehler, Alexander'
            ))
            cas.add(UceDynamicMetadata(key='date', value=item['date'], valueType='String', comment='The date this text was originally created or written by the user.'))
            cas.add(UceDynamicMetadata(key='domain', value=item['domain'], valueType='Enum', comment='The domain where this document belongs to.'))
            cas.add(UceDynamicMetadata(key='source', value=item['source'], valueType='Url', comment='Either the exact source url (if scraped) or the citation where this original human text was taken from.'))
            cas.add(UceDynamicMetadata(key='synthetization', value=self.__sanitize_string(json.dumps(item['synthetization'])), valueType='Json', 
                                        comment='The synthetization object of this text, containing a full AI rewrite of the text and also a chunk-based replacement somewhere in the middle of the text through AI. Can contain multiple AI agents.'))
            return cas.to_xmi(), file_name
        except Exception as ex:
            print(f"Couldn't parse the collected item with id {item['id']} to a XMI file.")
            print(ex)
            return None, None

    def __upload_xmi_to_uce(self, xmi_string, file_name, corpusId):
        try:
            url = "http://141.2.108.197:4567/api/ie/upload/uima"
            files = {
                "file": (file_name, io.StringIO(xmi_string), "application/xml"),
            }
            data = {
                "corpusId": corpusId
            }
            response = requests.post(url, files=files, data=data)
            if response.status_code == 200:
                print("Upload successful:", response.text)
            else:
                print(f"Upload failed with status {response.status_code}: {response.text}")
        except Exception as ex:
            print("Failed to upload XMI file:", ex)

    def synch_from_mongo_db_to_uce(self, mongo_conn_string, uceCorpusId):
        mongo_conn = MongoDBConnection(mongo_conn_string)
        collected_items = mongo_conn.get_collected_items_with_synth()

        # Let's do it in parallel.
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for item in collected_items:
                xmi_string, name = self.__build_xmi_from_collected_item(item)
                if xmi_string is None or name is None:
                    continue

                futures.append(executor.submit(self.__upload_xmi_to_uce, xmi_string, name, uceCorpusId))

            # Wait for all futures to complete
            concurrent.futures.wait(futures)

    def export_from_disc_to_disc(self, level, force=False):
        collector_paths = [ f.path for f in os.scandir(self.root_data_path) if f.is_dir() ]
        for col_path in collector_paths:
            input = os.path.join(col_path, level)
            if not os.path.exists(input):
                print(f'Input path: {input} not existing, skipping this collector then.')
                continue

            output = os.path.join(col_path, 'xmi_' + level)
            # If the path already exists, then we have already exported this collection on this leve.
            if os.path.exists(output) and not force:
                print('Skipping collection since already exported: ' + output)
                continue            
            self.counters[output] = 1
            os.makedirs(output, exist_ok=True)

            print(f'Exporting {input} to {output}')
            for filename in os.listdir(input):

                if filename.endswith('.json'): 
                    file_path = os.path.join(input, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        self.__export_df(pd.DataFrame(data), output, level)

if __name__ == '__main__':
    print('Starting the UIMA exporter.')
    base_data_path = os.getenv('DATA_ROOT_PATH')
    exporter = UIMAExporter(base_data_path)
    exporter.synch_from_mongo_db_to_uce(os.getenv('MONGO_DB_CONNECTION'), 34)
