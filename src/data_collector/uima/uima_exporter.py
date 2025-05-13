import os
import sys
import pathlib
import json
import pandas as pd
import uuid
import zipfile
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

    def __build_xmi_from_secondary_db_item(self, item, authors):
        try:
            if 'Manuel Schaaf' in authors:
                item['domain'] = 'PrismAI'

            id = str(uuid.uuid4())
            file_name = f'{item['domain']}_{id}'
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
                documentId=id, 
                documentUri='dataset_' + item['domain'] + item['id']
            ))
            cas.add(DocumentAnnotation(
                author=authors
            ))
            cas.add(UceDynamicMetadata(key='date', value=item['date'], valueType='Enum', comment="The date this text was released within the dataset (it's domain)."))
            cas.add(UceDynamicMetadata(key='language', value=item['lang'], valueType='Enum', comment='The language of this document.'))
            cas.add(UceDynamicMetadata(key='dataset', value=item['domain'], valueType='Enum', comment='The domain where this document belongs to.'))
            cas.add(UceDynamicMetadata(key='label', value=item['label'], valueType='Enum', comment="Either 'ai', 'human' or 'fusion'."))
            cas.add(UceDynamicMetadata(key='source', value=item['source'], valueType='String', comment='The original filename within the dataset this text belonged to.'))
            cas.add(UceDynamicMetadata(key='model', value=item['agent'], valueType='Enum', comment="The model this text was created by, if it's an AI-generated text."))
            return cas.to_xmi(), file_name
        except Exception as ex:
            print(f"Couldn't parse the collected item with id {item['id']} to a XMI file.")
            print(ex)
            return None, None

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
                author='Kevin Bönisch and Manuel Schaaf and Alexander Mehler'
            ))
            cas.add(UceDynamicMetadata(key='date', value=item['date'], valueType='String', comment='The date this text was originally created or written by the user.'))
            cas.add(UceDynamicMetadata(key='language', value=item['lang'], valueType='Enum', comment='The language of this document.'))
            cas.add(UceDynamicMetadata(key='domain', value=item['domain'], valueType='Enum', comment='The domain where this document belongs to.'))
            cas.add(UceDynamicMetadata(key='source', value=item['source'], valueType='Url', comment='Either the exact source url (if scraped) or the citation where this original human text was taken from.'))
            cas.add(UceDynamicMetadata(key='synthetization', value=self.__sanitize_string(json.dumps(item['synthetization'])), valueType='Json', 
                                        comment='The synthetization object of this text, containing a full AI rewrite of the text and also a chunk-based replacement somewhere in the middle of the text through AI. Can contain multiple AI agents.'))
            return cas.to_xmi(), file_name
        except Exception as ex:
            print(f"Couldn't parse the collected item with id {item['id']} to a XMI file.")
            print(ex)
            return None, None

    def __write_xmi_to_disc(self, xmi_string, file_name, zip_writer=None, base_path=None):
        """
        Writes the XMI string to a zip archive if `zip_writer` is provided, 
        otherwise writes to disk under base_path.
        """
        try:
            xmi_filename = f"{file_name}.xmi"
            if zip_writer is not None:
                zip_writer.writestr(xmi_filename, xmi_string)
            else:
                if base_path is None:
                    raise ValueError("base_path must be provided if not writing to a zip archive.")
                os.makedirs(base_path, exist_ok=True)
                file_path = os.path.join(base_path, xmi_filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(xmi_string)
        except Exception as ex:
            print(f"Failed to write {file_name}.xmi:", ex)

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
                pass
                #print("Upload successful:", response.text)
            else:
                print(f"Upload failed with status {response.status_code}: {response.text}")
        except Exception as ex:
            print("Failed to upload XMI file:", ex)

    def synch_secondary_dataset_to_uce(self, mongo_conn_string, uceCorpusId, dataset_name, authors):
        print('Doing ' + dataset_name)
        mongo_conn = MongoDBConnection(mongo_conn_string)
        collected_items = mongo_conn.get_secondary_dataset_items(dataset_name) 
        
        futures = []
        
        print('Phase 1:')
        # Phase 1: Process first 300 items with 1 worker
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for idx, item in enumerate(collected_items):
                xmi_string, name = self.__build_xmi_from_secondary_db_item(item, authors)
                if xmi_string is None or name is None:
                    continue
                
                futures.append(executor.submit(self.__upload_xmi_to_uce, xmi_string, name, uceCorpusId))
                
                if idx == 299:
                    break
        
        concurrent.futures.wait(futures)
        
        # Phase 2: Process remaining items with 20 workers
        print('Phase 2:')
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            for item in collected_items:  # Continue where we left off
                xmi_string, name = self.__build_xmi_from_secondary_db_item(item, authors)
                if xmi_string is None or name is None:
                    continue
                
                futures.append(executor.submit(self.__upload_xmi_to_uce, xmi_string, name, uceCorpusId))
        
        concurrent.futures.wait(futures)

    def synch_collected_items_to_uce(self, mongo_conn_string, uceCorpusId):
        print('Doing PrismAI Dataset')
        mongo_conn = MongoDBConnection(mongo_conn_string)
        collected_items = mongo_conn.get_collected_items_with_synth()
        
        futures = []
        
        # Phase 1: First 300 items with 1 worker
        print('Phase 2:')
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            for idx, item in enumerate(collected_items):
                xmi_string, name = self.__build_xmi_from_collected_item(item)
                if xmi_string is None or name is None:
                    continue
                
                futures.append(executor.submit(self.__upload_xmi_to_uce, xmi_string, name, uceCorpusId))
                
                if idx == 299:
                    break
        
        concurrent.futures.wait(futures)

        # Phase 2: Remaining items with 20 workers
        print('Phase 2:')
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            for item in collected_items:  # Continue with the rest
                xmi_string, name = self.__build_xmi_from_collected_item(item)
                if xmi_string is None or name is None:
                    continue
                
                futures.append(executor.submit(self.__upload_xmi_to_uce, xmi_string, name, uceCorpusId))
        
        concurrent.futures.wait(futures)

    def export_datasets_to_disc(self, mongo_conn_string, output_path, dataset_name, authors):
        print('Doing ' + dataset_name)
        mongo_conn = MongoDBConnection(mongo_conn_string)
        collected_items = mongo_conn.get_secondary_dataset_items(dataset_name)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for item in collected_items:
                futures.append(executor.submit(self.__build_xmi_from_secondary_db_item, item, authors))

            base_output_path = os.path.join(output_path, dataset_name)
            os.makedirs(base_output_path, exist_ok=True)

            file_counter = 0
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Exporting XMI files to disc"):
                xmi_string, file_name = future.result()
                if xmi_string and file_name:
                    file_counter += 1
                    batch_dir_number = ((file_counter - 1) // 1000 + 1) * 1000  # 1000, 2000, 3000, ...
                    batch_dir_path = os.path.join(base_output_path, str(batch_dir_number))
                    os.makedirs(batch_dir_path, exist_ok=True)

                    self.__write_xmi_to_disc(xmi_string, file_name, base_path=batch_dir_path)

if __name__ == '__main__':
    print('Starting the UIMA exporter.')
    base_data_path = os.getenv('DATA_ROOT_PATH')
    exporter = UIMAExporter(base_data_path)
    #exporter.synch_secondary_dataset_to_uce(os.getenv('MONGO_DB_CONNECTION'), 35, 'Ghostbuster', 'Vivek Verma, Eve Fleisig, Nicholas Tomlin, Dan Klein')
    #exporter.synch_secondary_dataset_to_uce(os.getenv('MONGO_DB_CONNECTION'), 35, 'CHEAT', 'Peipeng Yu, Jiahan Chen, Xuan Feng, Zhihua Xia')
    #exporter.synch_secondary_dataset_to_uce(os.getenv('MONGO_DB_CONNECTION'), 35, 'HC3-Plus', 'Zhenpeng Su, Xing Wu, Wei Zhou, Guangyuan Ma, Songlin Hu')
    #exporter.synch_secondary_dataset_to_uce(os.getenv('MONGO_DB_CONNECTION'), 35, 'SeqXGPT', 'Pengyu Wang, Linyang Li, Ke Ren, Botian Jiang, Dong Zhang, Xipeng Qiu')
    #exporter.synch_secondary_dataset_to_uce(os.getenv('MONGO_DB_CONNECTION'), 35, 'OpenLLMText', 'Yutian Chen, Hao Kang, Vivian Zhai, Liangze Li, Rita Singh, Bhiksha Raj')
    #exporter.synch_secondary_dataset_to_uce(os.getenv('MONGO_DB_CONNECTION'), 35, 'PrismAI', 'Manuel Schaaf, Kevin Bönisch, Alexander Mehler')
    #exporter.synch_collected_items_to_uce(os.getenv('MONGO_DB_CONNECTION'), 34)
    
    output = '/mnt/c/home/projects/prismAI/datasets/AIGT-World/xmi'
    exporter.export_datasets_to_disc(os.getenv('MONGO_DB_CONNECTION'), output, 'SeqXGPT', 'Pengyu Wang, Linyang Li, Ke Ren, Botian Jiang, Dong Zhang, Xipeng Qiu')
