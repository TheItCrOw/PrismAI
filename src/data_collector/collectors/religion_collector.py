import os
import json
import pandas as pd
import importlib

from collections import defaultdict
from datetime import datetime
from collected_item import CollectedItem
from data_collector.collector import Collector


class ReligionCollector(Collector):

    def __init__(self, root_path):
        super().__init__(root_path)

    def init(self):
        pass

    def read_english_bible(self):
        input, output, meta = self.get_paths()
        data = defaultdict(lambda: {'text': '', 'chunks': []})

        with open(os.path.join(input, 'en_bible.txt'), 'r', encoding='utf-8') as file:
            book, chapter = None, None
            for line in file:
                line = line.strip()
                if not line:
                    continue

                # Check if we have a new verse line or just a regular text line
                if ':' in line:
                    # Try splitting only the first occurrence of space after chapter:verse
                    parts = line.split(' ', 1)
                    if len(parts) < 2:
                        continue

                    chapter_verse, verse_text = parts
                    if ':' in chapter_verse:
                        try:
                            chapter_num, verse_num = map(int, chapter_verse.split(':'))
                        except ValueError:
                            continue

                        # update book and chapter
                        if chapter is None or chapter != chapter_num:
                            chapter = chapter_num
                        if book is None:
                            book = 'Genesis' # Default

                        key = (book, chapter)
                        data[key]['text'] += verse_text + ' '
                        data[key]['chunks'].append(f"{chapter_verse} {verse_text}")

        rows = []
        for (book, chapter), content in data.items():
            rows.append({
                'book': book,
                'chapter': chapter,
                'text': content["text"].strip(),
                'chunks': content["chunks"],
                'lang': 'en-EN'
            })

        return pd.DataFrame(rows)

    def read_german_bible(self):
        input, output, meta = self.get_paths()
        data = defaultdict(lambda: {'text': '', 'chunks': []})

        with open(os.path.join(input, 'de_bible.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                # split by space after the verse label (e.g., "Gen 1:1")
                parts = line.split(" ", 2)
                if len(parts) >= 3:
                    book = parts[0]
                    chapter_verse = parts[1].split(":")
                    text = parts[2].strip()
                    
                    # get chapter and verse
                    chapter = int(chapter_verse[0])
                    verse = int(chapter_verse[1])
                    
                    # gorup text and chunks by book and chapter
                    key = (book, chapter)
                    data[key]['text'] += text + ' '
                    data[key]['chunks'].append(f"{verse}: {text}") 

        rows = []
        for (book, chapter), content in data.items():
            rows.append({
                'book': book,
                'chapter': chapter,
                'text': content['text'].strip(),
                'chunks': content['chunks'],
                'lang': 'de-DE'
            })

        return pd.DataFrame(rows)

    def collect(self, force=False):
        super().collect('RELIGION')
        
        input, output, meta = self.get_paths()
        os.makedirs(output, exist_ok=True)

        total_counter = 0
        
        # Let's check if we already collected this data. Then we dont need to again
        if not force and os.path.exists(meta):
            self.read_meta_file()
            print(f'{self.meta['total_collected']} chapters were already collected at {self.meta['collected_at']}, hence we skip a redundant collection.')
            print("If you'd like to collect anyway, set the force variable to True.")
            return

        items = []
        bibles_df = pd.concat([self.read_german_bible(), self.read_english_bible()])

        for index, row in bibles_df.iterrows():
            item = CollectedItem(
                text=row['text'],
                chunks=row['chunks'],
                domain='religion',
                date='-',
                source='bible',
                lang=row['lang']
            )
            items.append(item)
            total_counter += 1

            if total_counter % 300 == 0 and total_counter != 0:
                df = pd.DataFrame([item.__dict__ for item in items])
                df.to_json(os.path.join(output, f"chapters_{total_counter // 300}.json"))
                print(f"Chapters collected: {total_counter}/{len(bibles_df)}")
                items = []

        df = pd.DataFrame([item.__dict__ for item in items])
        total_counter += len(df)
        df.to_json(os.path.join(output, "chapters_last.json"))
        print(f"Chapters collected:{total_counter}/{total_counter}")

        # Save some metadata about this collection
        self.write_meta_file(total_counter, datetime.now())
        print(f'Collection finished. Total Chapters collected: {total_counter}')

    def get_folder_path(self):
        return "religion"
