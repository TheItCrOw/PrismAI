#!/usr/bin/env python
# coding: utf-8

# # About
# 
# Here we collect our dataset for detecting AI-generated content. 
# 
# -------------------
# 
# A list of possible domains to fetch content from:
# 
# - Politics
#     - [German Bundestag](https://www.bundestag.de/) ✔
#     - [House of Commons](https://reshare.ukdataservice.ac.uk/854292/) ✔
# - Student/School
#     - [Kaggle Student Essays](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/data?select=train.csv) ✔
#     - Take the english dataset and translate it?
# - Research
#     - [Arxiv](https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=language+model&terms-0-field=all&classification-physics_archives=all&classification-include_cross_list=include&date-year=&date-filter_by=date_range&date-from_date=2005&date-to_date=2020&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first) ✔
#     - [I've seen a paper recently that creates full AI written papers. Maybe I can use it.](https://www.kaggle.com/discussions/general/527817#2958636)
# - News
#     - [Spiegel Online](https://www.spiegel.de/) ✔
#     - [CNN Articles](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail) ✔
# - Blogs/Tutorials/Forums
# - Law
#     - [German Open Legal Data](https://de.openlegaldata.io/) ✔
#     - [European Court of Human Rights Cases](https://www.kaggle.com/datasets/mathurinache/ecthrnaacl2021/data) ✔
# - Philosophy
#     - Gutenberg Project ([ENG](https://www.gutenberg.org/) | [GER](https://www.projekt-gutenberg.org/)) ✔
# - Literature
#     - Gutenberg Project ([ENG](https://www.gutenberg.org/) | [GER](https://www.projekt-gutenberg.org/)) ✔
# - Blogs, Food and Lifestyle
#     - [Food Blogs](https://detailed.com/food-blogs/)
#     - [WebBlogs](https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus?select=blogtext.csv) (ENG) ✔
# - Religion
#     - [Bible](https://github.com/mxw/grmr/blob/master/src/finaltests/bible.txt) (ENG|GER) ✔
# - Gaming
# 
# Interesting Languages:
# 
# - English
# - German

# In[13]:


import sys
import importlib
import os
import pandas as pd
import argparse

from IPython.display import Javascript
from dotenv import load_dotenv
from tqdm.notebook import tqdm

load_dotenv()


# In[14]:


class CONFIG:
    SRC_ROOT_PATH = os.getenv('SRC_ROOT_PATH')
    SRC_ROOT_PATH_COLL = os.getenv('SRC_ROOT_PATH_COLL')
    DATA_ROOT_PATH = os.getenv('DATA_ROOT_PATH')


# In[15]:


# So that it includes local imports. This is some next level python shit import
sys.path.insert(0, CONFIG.SRC_ROOT_PATH)
sys.path.insert(0, CONFIG.SRC_ROOT_PATH_COLL)


# In[16]:


import collector
import collected_item
import collectors.bundestag_collector
import collectors.house_of_commons_collector
import collectors.student_essays_collector
import collectors.arxiv_collector
import collectors.spiegel_collector
import collectors.cnn_news_collector
import collectors.open_legal_data_collector
import collectors.euro_court_cases_collector
import collectors.religion_collector
import collectors.gutenberg_collector
import collectors.blog_corpus_collector
import collectors.blog_corpus_collector

import data_collector.agents.ai_agent
import data_collector.agents.openai_agent

def reload():
    importlib.reload(collector)
    importlib.reload(collected_item)
    importlib.reload(collectors.bundestag_collector)
    importlib.reload(collectors.house_of_commons_collector)
    importlib.reload(collectors.student_essays_collector)
    importlib.reload(collectors.arxiv_collector)
    importlib.reload(collectors.spiegel_collector)
    importlib.reload(collectors.cnn_news_collector)
    importlib.reload(collectors.open_legal_data_collector)
    importlib.reload(collectors.euro_court_cases_collector)
    importlib.reload(collectors.religion_collector)
    importlib.reload(collectors.gutenberg_collector)
    importlib.reload(collectors.blog_corpus_collector)

    importlib.reload(data_collector.agents.ai_agent)
    importlib.reload(data_collector.agents.openai_agent)

reload()


# Check if we have cli arguments passed in. This is only done when this notebook is turned to a script.

# In[ ]:


run_as_script = False
parser = argparse.ArgumentParser(description="Worker Script for the orchestrator.")

parser.add_argument("--collectors", nargs="+", help="Pass in the collectors of this instance. Only works of --script-mode=True")

try:
    args = parser.parse_args()
    print(args.collectors)
    run_as_script = True
except:
    print('CLI parsing failed - this is hence run as a notebook.')


# ## Init the Collectors

# In[18]:


# If this is run as a script, that means the orcestrator passes in a list of collectors we are supposed to use.
# Fill the list dynamically then. If its from within a notebook, just add them by hand.
if run_as_script:
    collection = []
    for collector_name in args.collectors:
        try:
            module_name, class_name = collector_name.rsplit(".", 1)
            # get the class
            CollectorClass = getattr(eval(module_name), class_name)
            # Instantiate the collector
            collector_instance = CollectorClass(CONFIG.DATA_ROOT_PATH)
            collection.append(collector_instance)
        except AttributeError as e:
            print(f"Error creating collector '{collector_name}': {e}")
else:
    collection = [
        collectors.bundestag_collector.BundestagCollector(CONFIG.DATA_ROOT_PATH),
        collectors.house_of_commons_collector.HouseOfCommonsCollector(CONFIG.DATA_ROOT_PATH),
        collectors.student_essays_collector.StudentEssaysCollector(CONFIG.DATA_ROOT_PATH),
        collectors.arxiv_collector.ArxivCollector(CONFIG.DATA_ROOT_PATH),
        #collectors.spiegel_collector.SpiegelCollector(CONFIG.DATA_ROOT_PATH),
        #collectors.cnn_news_collector.CNNNewsCollector(CONFIG.DATA_ROOT_PATH),
        #collectors.open_legal_data_collector.OpenLegalDataCollector(CONFIG.DATA_ROOT_PATH),
        #collectors.euro_court_cases_collector.EuroCourtCasesCollector(CONFIG.DATA_ROOT_PATH),
        #collectors.religion_collector.ReligionCollector(CONFIG.DATA_ROOT_PATH),
        #collectors.gutenberg_collector.GutenbergCollector(CONFIG.DATA_ROOT_PATH),
        #collectors.blog_corpus_collector.BlogCorpusCollector(CONFIG.DATA_ROOT_PATH)
    ]

# Check the loaded collection
if run_as_script:
    print(f'Dynamically created collectors: {[type(c).__name__ for c in collection]}')
else:
    print(f'Manually defined collectors: {[type(c).__name__ for c in collection]}')


# In[6]:


total_items = 0

for coll in collection:
    try:
        coll.init()
        coll.collect()
        total_items += coll.get_count()
    except Exception as ex:
        print('ERROR: Current collection failed due to an error: ')
        print(ex)
        print('\n ***** Continuing with the other collectors. ***** \n')

print('\n\n ==================================================== \n\n')
print(f'All collectors finished. Total data items: {total_items}')


# # Generate the AI Content
# 
# After we've collected the dataset, we want to create it's AI-generated counterpart. We do this on multiple levels:
# 
# - Inject passages of AI-generated content
# - Trying to rewrite the whole text as an AI agent.
# - Trying different models?

# In[7]:


agents = [
    data_collector.agents.openai_agent.OpenAIAgent(name='gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
]


# Foreach agent, we go through all different collectors and synthesize the texts.

# In[8]:


take_per_collector = 250
force_synth = True


# In[ ]:


# Foreach collector (so foreach domain)
for coll in tqdm(collection, desc='Collectors', leave=False):
    items_count = 0
    df_count = 1
    items_dfs = coll.get_collected_items_dfs()
    collector_progress = tqdm(items_dfs, desc=f'Processing Collector Chunks of: {coll.get_folder_path().upper()}', leave=False)
    
    # We dont have a single huge dataframe file, but smaller chunks instead.
    for df in collector_progress:
        synth_items = []
        stored_path = os.path.join(coll.get_synthesized_path(), f"items_{df_count}.json")
        if not force_synth and os.path.exists(stored_path):
            # print('Skipping this chunk as it is already synthesized and stored. Use force=True otherwise.')
            df_count += 1
            continue

        if items_count >= take_per_collector:
            break

        for index, row in tqdm(df.iterrows(), desc="Current Chunk Items", total=len(df), leave=False):
            if items_count >= take_per_collector:
                break
            item = collected_item.CollectedItem.from_dict(row)

            # We have multiple agents. Foreach agent, synthesize the item
            for agent in agents:            
                try:
                    item.synthetization.append({
                        'agent': agent.name,
                        'synth_obj': agent.synthesize_collected_item(item, coll)
                    })
                    synth_items.append(item)
                    items_count += 1
                except Exception as ex:
                    print(f'There was an error with collector {coll.get_folder_path()} from source df chunk {df_count}')
                    print(ex)

        # Save synthesized items of that collected chunk.
        df = pd.DataFrame([item.__dict__ for item in synth_items])
        df.to_json(stored_path)
        df_count += 1


# In[ ]:




