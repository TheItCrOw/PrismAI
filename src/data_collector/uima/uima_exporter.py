import os
import pathlib
import json
import pandas as pd

from cassis import *

class UIMAExporter():
    ''''
    An exporter object, that exports the collected items into a UIMA Cas XMI file.
    '''

    def __init__(self, paths):
        self.paths = paths
        with open('UceDynamicMetadata.xml', 'rb') as f:
            self.typesystem = load_typesystem(f)

    def __export_df(self, df):
        for index, row in df.iterrows():
            cas = Cas(
                sofa_string = row['text'],
                document_language = row['lang'],
                typesystem = self.typesystem
            )
            UceDynamicMetadata = self.typesystem.get_type('org.texttechnologylab.annotation.uce.Metadata')
            cas.add(UceDynamicMetadata(key='domain', value=row['domain'], valueType='String'))
            cas.add(UceDynamicMetadata(key='source', value=row['source'], valueType='Url'))
            cas.add(UceDynamicMetadata(key='synthetization', value=json.dumps(row['synthetization']), valueType='Json'))
            cas.add(UceDynamicMetadata(key='feature_space', value=json.dumps(row['feature_space']), valueType='Json'))
            
            cas.to_xmi(f'export_{index}.xmi')

    def export(self):
        for path in self.paths:
            print(f'Doing path: {path}')
            for filename in os.listdir(path):

                if filename.endswith('.json'): 
                    file_path = os.path.join(path, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        self.__export_df(pd.DataFrame(data))

if __name__ == '__main__':
    # TODO: Finish this properly!
    print('Starting the UIMA exporter.')
    exporter = UIMAExporter(['/mnt/c/home/projects/prismAI/data/bundestag/feature_space/'])
    exporter.export()
