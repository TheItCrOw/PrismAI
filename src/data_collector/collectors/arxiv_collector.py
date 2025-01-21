import os
import json
import requests
import pandas as pd
import importlib
import unicodedata
import fitz
import string
import regex as re
import time

from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
from data_collector.collected_item import CollectedItem
from data_collector.collector import Collector


class ArxivCollector(Collector):
    search_url = "https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=&terms-0-field=all&classification-computer_science=y&classification-physics_archives=all&classification-include_cross_list=include&date-year=&date-filter_by=date_range&date-from_date=2005&date-to_date=2020&date-date_type=submitted_date&abstracts=hide&size={TAKE}&order=-announced_date_first&start={SKIP}"
    arxiv_base_url = "https://arxiv.org/pdf/"

    def __init__(self, root_path):
        super().__init__(root_path)

    def init(self):
        pass

    def clean_extracted_text(self, text):
        """
        Cleans weird UTF-8 characters from text. (Well, maybe we want to keep those actually?)
        """

        # Normalize text to combine characters into canonical form
        text = unicodedata.normalize("NFKC", text)

        allowed_chars = string.printable  # ASCII printable characters
        # extend to Unicode ranges
        allowed_chars += "".join(chr(i) for i in range(0x20, 0x7F))  # Basic Latin
        allowed_chars += "".join(
            chr(i) for i in range(0xA0, 0xD7FF)
        )  # Latin-1 Supplement and more

        # Remove characters not in that list.
        cleaned_text = "".join(c for c in text if c in allowed_chars)

        # Remove any lingering control characters or zero-width spaces
        cleaned_text = re.sub(r"[\u200B-\u200D\uFEFF]", "", cleaned_text)

        return cleaned_text.strip()

    def scrape(self, paper_amount=9000, skip=0):
        """'
        From the Arxiv page, scrapes the pdfs and puts them into the raw input directory
        """
        input, output, meta = self.get_collection_paths()
        os.makedirs(input, exist_ok=True)

        for i in range(1 + skip, skip + paper_amount):
            # 2001 means papers from January 2020. The id after the dot is just a counter
            pdf_url = self.arxiv_base_url + f"2001.{i:05d}"
            print(pdf_url)
            paper_id = pdf_url.split("/")[-1]  # Extract paper ID for naming
            file_name = os.path.join(input, f"{paper_id}.pdf")

            try:
                print(f"Downloading {pdf_url}")
                pdf_response = requests.get(pdf_url)
                pdf_response.raise_for_status()

                with open(file_name, "wb") as pdf_file:
                    pdf_file.write(pdf_response.content)
                print(f"Saved {file_name}")
                time.sleep(0.5)
            except requests.RequestException as e:
                print(f"Error downloading {pdf_url}: {e}")
                continue

    def extract_text_megaparse(self, pdf_file_path):
        """
        Extracing text from PDF to Markdown with MegaParse:
        https://github.com/QuivrHQ/MegaParse/tree/main
        Update: Yeah no, it's way too buggy. Appearntly it only works with python 3.11
        """
        pass

    def extract_text_fitz(self, pdf_file_path):
        try:
            # Open the PDF file
            with fitz.open(pdf_file_path) as pdf:
                text = ""
                pages = []
                # Iterate through all pages
                for page_num in range(len(pdf)):
                    page = pdf[page_num]
                    # All the two columned papers have TONS of \n which are just
                    # to break the line. We replace those with spaces, its more fitting.
                    page_text = page.get_text().replace("\n", " ")
                    pages.append(page_text)
                    text += page_text
            return text.strip(), pages
        except Exception as e:
            print(f"Error extracting text from {pdf_file_path}: {e}")
            return ""

    def collect(self, force=False):
        super().collect("ARXIV PAPERS")

        input, output, meta = self.get_collection_paths()
        os.makedirs(output, exist_ok=True)
        os.makedirs(input, exist_ok=True)

        total_files = len([name for name in os.listdir(input)])
        total_counter = 0

        # Let's check if we already collected this data. Then we dont need to again
        if not force and os.path.exists(meta):
            self.read_meta_file()
            print(
                f"{self.meta['total_collected']} papers were already collected at {self.meta['collected_at']}, hence we skip a redundant collection."
            )
            print("If you'd like to collect anyway, set the force variable to True.")
            return

        # We search for arxiv papers using the search_url above, then we scrape
        self.scrape(skip=1000)

        # If we're done scraping the papers, we need to extract the text and do our collector thing.
        counter = 1
        items = []
        for name in os.listdir(input):
            file_path = os.path.join(input, name)

            try:
                text, chunks = self.extract_text_fitz(file_path)
                if text == "":
                    print("Paper extraction gave empty text, skipping it.")
                    continue
                item = CollectedItem(
                    text=text,
                    chunks=chunks,
                    domain="arxiv_papers",
                    date="-",
                    source=self.arxiv_base_url + name.replace(".pdf", ""),
                    lang="en-EN",
                )
                items.append(item)
            except Exception as ex:
                print("Couldn't collect a paper, error: ")
                print(ex)
            print(f"Done with {counter}/{total_files} papers.")
            counter += 1

            if counter % 100 == 0 and counter != 0:
                df = pd.DataFrame([item.__dict__ for item in items])
                total_counter += len(df)
                df.to_json(
                    os.path.join(output, f"collected_papers_{counter // 100}.json")
                )
                items = []

        df = pd.DataFrame([item.__dict__ for item in items])
        total_counter += len(df)
        df.to_json(os.path.join(output, f"collected_papers_last.json"))
        print(f"Paper Dataframes collected: {counter}/{total_files}")

        # Save some metadata about this collection
        self.write_meta_file(total_counter, datetime.now())
        print(f"Collection finished. Total Papers collected: {total_counter}")

    def get_folder_path(self):
        return "arxiv_papers"
