import os

from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm
from luminar.encoder import LuminarEncoder
from datasets import Dataset, DatasetDict, ClassLabel
from unicodedata import normalize

from luminar.utils import get_best_device, get_matched_datasets


class DataPiepline:
    """
    Handles the processing, encoding and uploading of datasets to the Hugging Face Hub from the Datalake (MongoDB).
    """

    def __init__(self, mongodb_uri: str):
        self.mongodb_uri = mongodb_uri
        self.dataset = None

    def fetch(self, sources: list[str], limit: int = 999999999, skip: int = 0, mongo_filter: Dict = None):
        """
        Fetches the given sources from the MongoDB.
        :param sources: List of dataset names to be included in the pipeline as outlined in the mongodb.
        :param limit: Limit the number of documents to fetch from each source. Default is infinity.
        :param skip: Number of documents to skip from the start of each source. Default is 0.
        :param mongo_filter: Optional filter to apply to the MongoDB query.
        """
        print("Fetching data from source: ", sources)
        client = MongoClient(self.mongodb_uri)
        db = client["prismai"]
        collected_docs = []

        with client.start_session() as session:
            for source in sources:
                collection = db.get_collection(f"dataset_{source}")
                print(f"Fetching from collection: {source}")

                total_docs = min(limit, collection.count_documents(mongo_filter, session=session))
                cursor = (collection
                          .find(mongo_filter, session=session, no_cursor_timeout=True, batch_size=100)
                          .limit(limit)
                          .sort([("_id", -1)])
                          .skip(skip))

                try:
                    for doc in tqdm(cursor, total=total_docs, desc=f"Reading {source}"):
                        doc.pop("_id", None)
                        # For arrow, everything needs to be uniform datatypes
                        doc_str = {k: str(v) for k, v in doc.items()}
                        collected_docs.append(doc_str)
                finally:
                    cursor.close()

        # Convert to Hugging Face dataset
        print("Converting to Hugging Face dataset...")
        self.dataset = Dataset.from_list(collected_docs)
        return self

    def encode(self, model: str, max_len: int = 512, batch_size: int = 512):
        if self.dataset is None:
            raise ValueError("Dataset is not set. Please fetch the dataset first.")

        device = get_best_device()
        print(f"Encoding dataset with model: {model} on device {device}")

        encoder = LuminarEncoder(model_name_or_path=model, max_len=max_len, device=device)

        # Tokenization stage
        self.dataset = self.dataset.map(
            lambda inputs: encoder.tokenize(inputs, truncate=False),
            input_columns=["text"],
            batched=True,
            batch_size=1024,  # Use a larger batch size for tokenization
            desc="Tokenizing"
        )

        # Optional: sort by sequence length if useful for batching
        if "length" in self.dataset.column_names:
            self.dataset = self.dataset.sort("length")

        # Encoding stage
        self.dataset = self.dataset.map(
            lambda inputs: encoder.rolling_process(inputs, max_chunks=5),
            batched=True,
            batch_size=batch_size,
            desc="Encoding",
            remove_columns=["input_ids", "attention_mask"],
            num_proc=1
        ).map(lambda inputs: {"feature_length": len(inputs["features"])})
        # self.dataset.select(range(1000)).to_json("test_dataset.json", orient="records", lines=True)

        return self

    def upload_to_hf(self, hf_token: str, full_name: str, split_name: str = "train"):
        self.dataset.push_to_hub(
            repo_id=full_name,
            token=hf_token,
            split=split_name,
            private=True,
        )

    def run(self,
            agent: str,
            source: str,
            hf_token: str,
            organization: str = "liberi-luminaris",
            fetch_filter: Dict = None,
            upload_non_encoded: bool = True,
            upload_encoded: bool = True,
            hf_name_prefix: str = None,
            split_name: str = "train"
            ):
        params = locals()
        print("New pipeline run with parameters:")
        for name, value in params.items():
            print(f"  {name}: {value}")

        print(self.fetch(sources=[source], mongo_filter=fetch_filter).dataset)
        if upload_non_encoded:
            self.upload_to_hf(hf_token=hf_token,
                              split_name=split_name,
                              full_name=f"{organization}/{source}{f"-{hf_name_prefix}" if hf_name_prefix else ""}")

        print(self.encode(model=f"{agent}", max_len=512, batch_size=128).dataset)
        if upload_encoded:
            if "/" in agent:
                agent = agent.split("/")[1]
            self.upload_to_hf(hf_token=hf_token,
                              split_name=split_name,
                              full_name=f"{organization}/{source}{f"-{hf_name_prefix}" if hf_name_prefix else ""}-encoded-{agent}")
        return self.dataset


if __name__ == "__main__":
    # Example usage:
    load_dotenv()
    pipeline = DataPiepline(mongodb_uri=os.getenv("MONGO_DB_CONNECTION"))

    # tiiuae/falcon-7b
    pipeline.run(agent="tiiuae/falcon-7b",
                 source="PrismAI_v2",
                 upload_non_encoded=False,
                 upload_encoded=True,
                 organization="TheItCrOw",
                 fetch_filter={"domain": "arxiv_papers"},
                 split_name="arxiv_papers",
                 hf_token=(Path.home() / ".hf_token").read_text().strip())
