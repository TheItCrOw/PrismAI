import os
import pathlib
import json
import pandas as pd
import uuid
import re

from dotenv import load_dotenv
from tqdm.notebook import tqdm
from cassis import *

load_dotenv()


class UIMAExporter:
    """'
    An exporter object, that exports the collected items into a UIMA Cas XMI file.
    """

    def __init__(self, root_data_path):
        self.root_data_path = root_data_path
        self.counters = {}
        self.__load_typesystems()

    def __sanitize_string(self, value):
        if isinstance(value, str):
            # Remove invalid XML characters
            value = re.sub(r"[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD]", "", value)
        return value

    def __load_typesystems(self):
        # This looks kinda ugly, but from the documentation, I don't know wny other way to load
        # the typesystems...
        with open("DocumentMetadata.xml", "rb") as f:
            metadata_ts = load_typesystem(f)

        with open("TextTechnologyDokumentAnnotation.xml", "rb") as f:
            doc_annotation_ts = load_typesystem(f)

        with open("UceDynamicMetadata.xml", "rb") as f:
            uce_ts = load_typesystem(f)

        self.typesystem = merge_typesystems(metadata_ts, doc_annotation_ts, uce_ts)
        print("======= Loaded UIMA Typesystems:")
        for t in self.typesystem.get_types():
            print(t)

    def __export_df(self, df, output_dir, level):
        for index, row in df.iterrows():
            try:
                if row["text"] == None or row["text"] == "":
                    continue

                file_name = f"{row['domain']}_{self.counters[output_dir]}.xmi"
                output_file = os.path.join(output_dir, file_name)
                cas = Cas(
                    sofa_string=self.__sanitize_string(row["text"]),
                    document_language=row["lang"],
                    typesystem=self.typesystem,
                )
                UceDynamicMetadata = self.typesystem.get_type(
                    "org.texttechnologylab.annotation.uce.Metadata"
                )
                DocumentAnnotation = self.typesystem.get_type(
                    "org.texttechnologylab.annotation.DocumentAnnotation"
                )
                DocumentMetadata = self.typesystem.get_type(
                    "de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData"
                )

                cas.add(
                    DocumentMetadata(
                        documentTitle=file_name,
                        documentId=str(uuid.uuid4()),
                        documentUri=output_file,
                    )
                )
                cas.add(
                    DocumentAnnotation(
                        author="Bönisch, Kevin and Stoeckel, Manuel and Mehler, Alexander"
                    )
                )
                cas.add(
                    UceDynamicMetadata(
                        key="date",
                        value=row["date"],
                        valueType="String",
                        comment="The date this text was originally created or written by the user.",
                    )
                )
                cas.add(
                    UceDynamicMetadata(
                        key="domain",
                        value=row["domain"],
                        valueType="Enum",
                        comment="The domain where this document belongs to.",
                    )
                )
                cas.add(
                    UceDynamicMetadata(
                        key="source",
                        value=row["source"],
                        valueType="Url",
                        comment="Either the exact source url (if scraped) or the citation where this original human text was taken from.",
                    )
                )
                cas.add(
                    UceDynamicMetadata(
                        key="synthetization",
                        value=self.__sanitize_string(json.dumps(row["synthetization"])),
                        valueType="Json",
                        comment="The synthetization object of this text, containing a full AI rewrite of the text and also a chunk-based replacement somewhere in the middle of the text through AI.",
                    )
                )

                if level == "feature_space":
                    cas.add(
                        UceDynamicMetadata(
                            key="feature_space",
                            value=json.dumps(row["feature_space"]),
                            valueType="Json",
                        )
                    )

                cas.to_xmi(output_file)
                self.counters[output_dir] += 1
            except Exception as ex:
                print("There was an error with one file, skipping it.")
                print(ex)

    def export(self, level, force=False):
        collector_paths = [
            f.path for f in os.scandir(self.root_data_path) if f.is_dir()
        ]
        for col_path in collector_paths:
            input = os.path.join(col_path, level)
            if not os.path.exists(input):
                print(
                    f"Input path: {input} not existing, skipping this collector then."
                )
                continue

            output = os.path.join(col_path, "xmi_" + level)
            # If the path already exists, then we have already exported this collection on this leve.
            if os.path.exists(output) and not force:
                print("Skipping collection since already exported: " + output)
                continue
            self.counters[output] = 1
            os.makedirs(output, exist_ok=True)

            print(f"Exporting {input} to {output}")
            for filename in os.listdir(input):
                if filename.endswith(".json"):
                    file_path = os.path.join(input, filename)
                    with open(file_path, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        self.__export_df(pd.DataFrame(data), output, level)


if __name__ == "__main__":
    print("Starting the UIMA exporter.")
    base_data_path = os.getenv("DATA_ROOT_PATH")
    exporter = UIMAExporter(base_data_path)
    exporter.export("synthesized", force=True)
