import pandas as pd
import glob
import statistics
import spacy
import torch
import json
import os
from detector import Detector
from causal_lm import CausalLM
import warnings

warnings.filterwarnings('ignore')
dataset_path = '/home/staff_homes/kboenisc/home/prismAI/data/'

def get_essays(label, skip, take):
    all_files = glob.glob(os.path.join(dataset_path, "*.csv"))
    df = None
    for filename in all_files:
        df = pd.concat([df, pd.read_csv(filename)[['text', 'label']]])
    print(f'Found {len(df)} essays.')

    # As a first step, we want to inject AI written sentences into human essays
    # and see if we can sort them out later again
    # 1 = AI generated, 0 = human written
    filtered_df = df[df['label'] == label].sort_index().iloc[skip: skip + take]
    filtered_essays = filtered_df['text'].tolist()
    print(f'Processing {len(filtered_essays)} filtered essays.')
    return filtered_essays

def preprocess_dataset():
    '''We want to preprocess the dataset'''
    nlp = spacy.load("en_core_web_sm")
    model = CausalLM('gpt2')

    filtered_essays = get_essays(1, 100)

    frequence = 5
    confidences = []
    for essay in filtered_essays:
        # Let's replace every X sentence
        doc = nlp(essay)
        sentences = [sent.text for sent in doc.sents]
        processed_essay = []
        for (i, sentence) in enumerate(sentences):
            if((i + 1) % frequence == 0):
                # Here we generate the sentence by LLM instead
                # We want the new text to be the same length if possible.
                input_text = ' '.join(processed_essay[-4:]).replace('[AI_FILLED]', '')
                output = model.generate_k_with_probs(
                    input_text, 
                    32)
                
                generated = output['generated_text']
                # We only want the first sentence from it
                llm_doc = nlp(generated)
                processed_essay.append(f'[AI_FILLED]{list(llm_doc.sents)[0].text}[AI_FILLED]')
                
                # We also want to track the confidences
                for step in output['steps']:
                    confidences.append(step['token_prob'].item())
            else:
                processed_essay.append(sentence)
        print(' '.join(processed_essay).strip())
        print('============\n\n')

    print(f'Smallest confidence: ${min(confidences)}')
    print(f'Biggest confidence: ${max(confidences)}')
    print(f'Average confidence: ${statistics.mean(confidences)}')


def generate_data(label, amount, ensemble):
    # https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
    detector = Detector(models=ensemble)
    essays = get_essays(label, 4000, amount)
    results = []
    # Let the detector do the work
    essays_avg = []
    counter = 0
    for essay in essays:
        try:
            if(len(essay) < 500):
                continue
            output = detector.detect(essay, sample_sequence_length=32)
            if(output == None):
                continue
            results.append(output)
            avg = output['ensemble_results'][0]['avg_prob']
            print(f'\n========== AVG: {avg} ==========\n')
            essays_avg.append(avg)
            print("Done with: " + str(counter))
            counter += 1
        except Exception as ex:
            print('Skipped one essay.')
            print(ex)
    print(statistics.mean(essays_avg))

    # At the end, delete the model
    del detector
    torch.cuda.empty_cache()
    print("Deleted detector")

    with open(f'outputs/generator_{label}_{ensemble[0].replace("/", "_")}_{amount}_outputs.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)

def test_detector(ensemble):
    detector = Detector(models=ensemble)
    essays = get_essays(0, 4200, 1000)
    counter = 0
    for essay in essays:
        if(len(essay) < 500):
            continue
        try:
            output = detector.detect(essay, sample_sequence_length=32)
            if(output == None):
                continue
            print(output['metadata']['pred'])
        except Exception as ex:
            print(ex)
        counter += 1


if __name__ == '__main__':
    ensemble = ['gpt2', 'mistralai/Mistral-7B-v0.1', 'daryl149/llama-2-7b-chat-hf']
    ensemble = ['google/gemma-2b']
    for model in ensemble:
        generate_data(0, 6000, [model])
        generate_data(1, 6000, [model])

