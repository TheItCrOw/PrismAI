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

# In[1]:


import sys
import importlib
import os
import pandas as pd
import time
import argparse

from dotenv import load_dotenv
from tqdm.notebook import tqdm

load_dotenv()


# In[2]:


class CONFIG:
    SRC_ROOT_PATH = os.getenv('SRC_ROOT_PATH')
    SRC_ROOT_PATH_COLL = os.getenv('SRC_ROOT_PATH_COLL')
    DATA_ROOT_PATH = os.getenv('DATA_ROOT_PATH')

    COLLECT = False
    SYNTHESIZE = False
    EXTRACT_FEATURES = True

    CAUSAL_LLMS = []


# In[3]:


# So that it includes local imports. This is some next level python shit import
sys.path.insert(0, CONFIG.SRC_ROOT_PATH)
sys.path.insert(0, CONFIG.SRC_ROOT_PATH_COLL)


# In[4]:


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

# In[5]:


synthesize_per_collector = 0
extract_features_per_collector = 15
skip_per_collector = 0
force_synth = False
force_featured = True
run_as_script = False


# In[ ]:


parser = argparse.ArgumentParser(description="Worker Script for the orchestrator.")

parser.add_argument("--collectors", nargs="+", help="Pass in the collectors of this instance. Only works if --script-mode=True")
parser.add_argument("--collect", action="store_true", help="Determine if we collect a dataset.")
parser.add_argument("--synth", action="store_true", help="Determine if we synthesize a dataset.")

parser.add_argument("--take", type=int, default=100, help="Determine how many items we synthesize per collector.")
parser.add_argument("--skip", type=int, default=0, help="Determine how many items we skip per collector.")

parser.add_argument("--featured", action="store_true", help="Determine if we feature_exract a dataset.")
parser.add_argument("--causal_llms", help="Pass in the CausalLLMs to generate the features.")

parser.add_argument("--force", action="store_true", help="Determine if we force a synthesization.")

try:
    args = parser.parse_args()
    print('=== Passed Parameters:')
    print('= Collectors:')
    print(args.collectors)

    print(f'= Take: {str(args.take)}')
    extract_features_per_collector = args.take
    synthesize_per_collector = args.take
    
    print(f'= Skip: {str(args.skip)}')
    skip_per_collector = args.skip
    
    print(f'= Force: {str(args.force)}')
    force_synth = args.force
    force_featured = args.force
    
    print(f'= Activated Levels:')
    print(f'- Collect: {args.collect}')
    CONFIG.COLLECT = args.collect
    print(f'- Synthesize: {args.synth}')
    CONFIG.SYNTHESIZE = args.synth
    print(f'- Extract features: {args.featured}')
    print(f'- Used CausalLLMs: {args.causal_llms}')
    CONFIG.CAUSAL_LLMS = args.causal_llms.split(',')
    CONFIG.EXTRACT_FEATURES = args.featured
    
    run_as_script = True
except:
    print('CLI parsing failed - this is hence run as a notebook.')


# ---------------------
# 
# # Collect
# 
# First, we gotta create a dataset and for that, we collect texts from various domains.

# In[7]:


# If this is run as a script, that means the orcestrator passes in a list of collectors we are supposed to use.
# Fill the list dynamically then. If its from within a notebook, just add them by hand.
if run_as_script:
    collection = []
    for collector_name in args.collectors:
        try:
            module_name, class_name = collector_name.rsplit(".", 1)
            CollectorClass = getattr(eval(module_name), class_name)
            collector_instance = CollectorClass(CONFIG.DATA_ROOT_PATH)
            collection.append(collector_instance)
        except AttributeError as e:
            print(f"Error creating collector '{collector_name}': {e}")
else:
    collection = [
        collectors.bundestag_collector.BundestagCollector(CONFIG.DATA_ROOT_PATH),
        #collectors.house_of_commons_collector.HouseOfCommonsCollector(CONFIG.DATA_ROOT_PATH),
        #collectors.student_essays_collector.StudentEssaysCollector(CONFIG.DATA_ROOT_PATH),
        #collectors.arxiv_collector.ArxivCollector(CONFIG.DATA_ROOT_PATH),
        #collectors.spiegel_collector.SpiegelCollector(CONFIG.DATA_ROOT_PATH),
        #collectors.cnn_news_collector.CNNNewsCollector(CONFIG.DATA_ROOT_PATH),
        #collectors.open_legal_data_collector.OpenLegalDataCollector(CONFIG.DATA_ROOT_PATH),
        #collectors.euro_court_cases_collector.EuroCourtCasesCollector(CONFIG.DATA_ROOT_PATH),
        #collectors.religion_collector.ReligionCollector(CONFIG.DATA_ROOT_PATH),
        #collectors.gutenberg_collector.GutenbergCollector(CONFIG.DATA_ROOT_PATH),
        #collectors.blog_corpus_collector.BlogCorpusCollector(CONFIG.DATA_ROOT_PATH)
    ]

if run_as_script:
    print(f'Dynamically created collectors: {[type(c).__name__ for c in collection]}')
else:
    print(f'Manually defined collectors: {[type(c).__name__ for c in collection]}')


# In[8]:


if CONFIG.COLLECT:
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
else:
    print('Collecting turned off - skipping it.')


# ---------------------
# 
# # Synthesize
# 
# After we've collected the dataset, we want to create it's AI-generated counterpart. We do this on multiple levels:
# 
# - Inject passages of AI-generated content
# - Trying to rewrite the whole text as an AI agent.
# - Trying different models

# In[9]:


agents = [
    data_collector.agents.openai_agent.OpenAIAgent(name='gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
]


# Foreach agent, we go through all different collectors and synthesize the texts.

# In[10]:


if CONFIG.SYNTHESIZE:
    # Foreach collector (so foreach domain)
    for coll in tqdm(collection, desc='Collectors', leave=False):
        items_count = 0
        df_count = 1
        items_dfs = coll.get_collected_items_dfs()
        collector_progress = tqdm(items_dfs, desc=f'Processing Collector Chunks of: {coll.get_folder_path().upper()}', leave=False)
        
        # We dont have a single huge dataframe file, but smaller chunks instead.
        for in_df in collector_progress:
            start = time.time()
            synth_items = []
            stored_path = os.path.join(coll.get_synthesized_path(), f"items_{df_count}.json")
            if not force_synth and os.path.exists(stored_path):
                print('Skipping this chunk as it is already synthesized and stored. Use force=True otherwise.')
                df_count += 1
                continue

            if items_count >= synthesize_per_collector:
                break

            for index, row in tqdm(in_df.iterrows(), desc="Current Chunk Items", total=len(in_df), leave=False):
                if items_count >= synthesize_per_collector:
                    break
                
                # If we have a skip parameter, skip as long as we need to.
                if skip_per_collector > 0 and items_count <= skip_per_collector:
                    items_count += 1
                    continue
                
                item = collected_item.CollectedItem.from_dict(row)

                # We have multiple agents. Foreach agent, synthesize the item
                for agent in agents:            
                    try:
                        item.synthetization.append({
                            'agent': agent.name,
                            'synth_obj': agent.synthesize_collected_item(item, coll)
                        })
                        synth_items.append(item)
                    except Exception as ex:
                        print(f'There was an error with collector {coll.get_folder_path()} from source df chunk {df_count}')
                        print(ex)
                items_count += 1

            # Save synthesized items of that collected chunk.
            out_df = pd.DataFrame([item.__dict__ for item in synth_items])
            out_df.to_json(stored_path)
            df_count += 1
            end = time.time()
            if run_as_script:
                print(f'Stored and wrote {len(out_df)} items to file {stored_path}')
                print(f'Total items done: {str(items_count)}')
                print(f'Chunk synthesization time: {str(end - start)}')

    print('Done with synthesization.')
else:
    print('Synthesization turned off - skipping it.')


# ---------------------
# 
# # Feature Extraction
# 
# After we have collected and synthesized a dataset, we can now start collecting the feature space. We do that with the Detector class, give him an ensemble of models and let it extract features.

# In[11]:


import detector
import causal_lm

importlib.reload(detector)
importlib.reload(causal_lm)


# Create the detector, which we use for feature extractio and also for prediction if we wanted (currently not). Also test the detection, which is also the feature extraction.
# 
# I want to try out the following models maybe:
# - [openGPT-X/Teuken-7B-instruct-research-v0.4](https://huggingface.co/meta-llama/Llama-3.2-3B)
# - GPT2 as a cheap alternative maybe
# - [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)

# In[ ]:


if CONFIG.EXTRACT_FEATURES:
    if run_as_script:
        det = detector.Detector(models=CONFIG.CAUSAL_LLMS, min_token_length=52)
    else:
        det = detector.Detector(models=['GPT2'], min_token_length=52)

    test_text = "In his first State of the Union address, President Joe Biden delivered an optimistic outlook on the nation's economy, celebrating its remarkable rebound while drawing a clear contrast with the policies of his predecessor, Donald Trump. Against the backdrop of ongoing challenges posed by the pandemic and geopolitical tensions, Biden sought to instill confidence in the American people and reaffirm his administration's commitment to fostering inclusive growth and resilience."
    print(det.detect(test_text))


# If everything works, we can go through the collectors and extract the features from each text.

# In[13]:


if CONFIG.EXTRACT_FEATURES:
    for coll in tqdm(collection, desc='Collectors', leave=False):
        items_count = 0
        df_count = 1
        items_dfs = coll.get_synthesized_items_dfs()
        collector_progress = tqdm(items_dfs, desc=f'Processing Synthesized Chunks of: {coll.get_folder_path().upper()}', leave=False)
        
        for in_df in collector_progress:
            start = time.time()
            featured_items = []
            stored_path = os.path.join(coll.get_feature_space_path(), f"items_{df_count}.json")
            if not force_featured and os.path.exists(stored_path):
                print('Skipping this chunk as it is already featured and stored. Use force=True otherwise.')
                df_count += 1
                continue

            if items_count >= extract_features_per_collector:
                break

            for index, row in tqdm(in_df.iterrows(), desc="Current Chunk Items", total=len(in_df), leave=False):
                if items_count >= extract_features_per_collector:
                    break
                
                # If we have a skip parameter, skip as long as we need to.
                if skip_per_collector > 0 and items_count <= skip_per_collector:
                    items_count += 1
                    continue
                
                item = collected_item.CollectedItem.from_dict(row)

                # Extract features from this item here.
                try:
                    # Do the original human text first.
                    fs_1 = det.detect(item.text, 
                                      sample_rate=24, 
                                      sample_sequence_length=48,
                                      k=100)
                    # The detect() method returns the fulltext, but we dont need to store that.
                    fs_1['metadata']['full_text'] = ''
                    item.feature_space = fs_1

                    # Then the synth objects
                    if item.synthetization is not None:
                        # We have multiple AI agents that did a synthesization
                        for agent_obj in item.synthetization:
                            synth_obj = agent_obj['synth_obj']
                            if synth_obj is None:
                                continue
                            # We did synthesize chunks and fulltexts.
                            chunks = synth_obj['synth_chunks']
                            fulltext = synth_obj['synth_fulltext']
                            if chunks is not None:
                                fs_2 = det.detect(chunks['synth_text'], 
                                      sample_rate=24, 
                                      sample_sequence_length=48,
                                      k=100)
                                fs_2['metadata']['full_text'] = ''
                                chunks['feature_space'] = fs_2 
                            if fulltext is not None:
                                fs_3 = det.detect(fulltext['synth_text'], 
                                      sample_rate=24, 
                                      sample_sequence_length=48,
                                      k=100)
                                fs_3['metadata']['full_text'] = ''
                                fulltext['feature_space'] = fs_3
                    featured_items.append(item)
                except Exception as ex:
                    print('Error while extracting features from an item:')
                    print(ex)

                items_count += 1

            # Save synthesized items of that collected chunk.
            out_df = pd.DataFrame([item.__dict__ for item in featured_items])
            out_df.to_json(stored_path)
            df_count += 1
            end = time.time()
            if run_as_script:
                print(f'Stored and wrote {len(out_df)} featured items to file {stored_path}')
                print(f'Total featured items done: {str(items_count)}')
                print(f'Chunk featured time: {str(end - start)}')

    print('Done with feature extraction.')
else:
    print('Synthesization turned off - skipping it.')


# In[ ]:




