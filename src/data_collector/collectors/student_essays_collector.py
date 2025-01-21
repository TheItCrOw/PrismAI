import os
import json
import pandas as pd
import importlib

from datetime import datetime
from data_collector.collected_item import CollectedItem
from data_collector.collector import Collector


class StudentEssaysCollector(Collector):
    def __init__(self, root_path):
        super().__init__(root_path)

    def init(self):
        pass

    def collect(self, force=False):
        super().collect("STUDENT ESSAYS")

        input, output, meta = self.get_collection_paths()
        os.makedirs(output, exist_ok=True)

        # we iterate through the exported essays and get the their texts
        total_files = len([name for name in os.listdir(input)])
        total_counter = 0

        # Let's check if we already collected this data. Then we dont need to again
        if not force and os.path.exists(meta):
            self.read_meta_file()
            print(
                f"{self.meta['total_collected']} essays were already collected at {self.meta['collected_at']}, hence we skip a redundant collection."
            )
            print("If you'd like to collect anyway, set the force variable to True.")
            return

        counter = 1
        for name in os.listdir(input):
            items = []
            file_path = os.path.join(input, name)
            in_df = pd.read_csv(file_path, low_memory=False)

            # 1 means is AI Generated, 0 means humman essays
            for index, row in in_df[in_df["label"] == 0].iterrows():
                essay = row["text"]
                chunks = essay.split("\n")
                try:
                    item = CollectedItem(
                        text=essay,
                        chunks=chunks,
                        domain="student_essays",
                        date="-",
                        source="https://www.kaggle.com/datasets/thedrcat/daigt-proper-train-dataset",
                        lang="en-EN",
                    )
                    items.append(item)
                except Exception as ex:
                    print("Couldn't collect an essay, error: ")
                    print(ex)

            df = pd.DataFrame([item.__dict__ for item in items])
            total_counter += len(df)
            df.to_json(os.path.join(output, str(counter) + ".json"))
            print(f"Essay Dataframes collected: {counter}/{total_files}")
            counter += 1

        # Save some metadata about this collection
        self.write_meta_file(total_counter, datetime.now())

        print(f"Collection finished. Total Essays collected: {total_counter}")

    def get_folder_path(self):
        return "student_essays"
