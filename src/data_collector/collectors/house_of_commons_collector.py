import os
import json
import pandas as pd
import importlib

from datetime import datetime
from data_collector.collected_item import CollectedItem
from data_collector.collector import Collector


class HouseOfCommonsCollector(Collector):
    def __init__(self, root_path):
        super().__init__(root_path)

    def init(self):
        pass

    def collect(self, force=False):
        super().collect("HOUSE OF COMMONS")

        input, output, meta = self.get_collection_paths()
        os.makedirs(output, exist_ok=True)

        # we iterate through the exported speeches and get the their texts
        total_files = len([name for name in os.listdir(input)])
        total_counter = 0

        # Let's check if we already collected this data. Then we dont need to again
        if not force and os.path.exists(meta):
            self.read_meta_file()
            print(
                f"{self.meta['total_collected']} speeches were already collected at {self.meta['collected_at']}, hence we skip a redundant collection."
            )
            print("If you'd like to collect anyway, set the force variable to True.")
            return

        counter = 1
        for name in os.listdir(input):
            items = []
            file_path = os.path.join(input, name)
            in_df = pd.read_csv(file_path, low_memory=False)

            def safe_join(sequence):
                return " ".join(map(str, sequence))

            # So they way these csv files are structured is basically a single sentence or
            # two is its own entry, but they are grouped by a subsection id which is maybe
            # a common topic? So the whole text is a whole topic and each chunk is a line.
            result_df = (
                in_df.groupby("subsection_id")
                .agg(body=("body", lambda x: safe_join(x)), chunks=("body", list))
                .reset_index()
            )

            for index, row in result_df.iterrows():
                try:
                    item = CollectedItem(
                        text=row["body"],
                        chunks=row["chunks"],
                        domain="house_of_commons",
                        date=name.replace(".csv", ""),
                        source="https://reshare.ukdataservice.ac.uk/854292/",
                        lang="en-EN",
                    )
                    items.append(item)
                except Exception as ex:
                    print("Couldn't collect a speech, error: ")
                    print(ex)

            df = pd.DataFrame([item.__dict__ for item in items])
            total_counter += len(df)
            df.to_json(os.path.join(output, str(counter) + ".json"))
            print(f"Years collected: {counter}/{total_files}")
            counter += 1

        # Save some metadata about this collection
        self.write_meta_file(total_counter, datetime.now())

        print(f"Collection finished. Total Speeches collected: {total_counter}")

    def get_folder_path(self):
        return "house_of_commons"
