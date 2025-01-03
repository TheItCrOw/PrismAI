import random
import hashlib
import traceback
import sys
import statistics

from causal_lm import CausalLM
from datetime import datetime
from luminar.luminar_model import Luminar

class Detector():
    '''A class that tries to identify AI texts'''

    def __init__(self, 
                 models=['gpt2'],
                 min_token_length = 32,
                 luminar_model_path = None):
        '''
        Possible models to choose from:
        daryl149/llama-2-7b-chat-hf
        gpt2
        mistralai/Mistral-7B-v0.1
        '''
        self.llm_ensemble = [CausalLM(m, include_spacy=False) for m in models]
        self.min_token_length = min_token_length
        self.luminar = None
        if luminar_model_path is not None:
            self.luminar = Luminar('/home/staff_homes/kboenisc/home/prismAI/PrismAI/src/luminar/models/luminar_llama2_4k.pth')
        print(f'Model ensemble: {[x.model_name for x in self.llm_ensemble]}\n')

    def detect(self, 
               text, 
               sample_rate=6,
               sample_sequence_length=12,
               k=10,
               temp=0.9,
               seed=42):
        '''A function that tries to detect AI sequences in a text'''

        random.seed(seed)
        ensemble_results = []
        # We have an ensemble of models which we use to detect
        for model in self.llm_ensemble:
            model_result = {
                'model': model.model_name,
                'sample_results': []
            }

            # Tokenize the text
            token_idx = model.tokenizer.encode(text, return_tensors='pt')
            # I dont think we can handle very short texts. We will see.
            if(len(token_idx[0]) < self.min_token_length):
                raise Exception(f'The text must be at least {self.min_token_length} tokens long, got only {len(token_idx[0])}.')

            # We will randomly sample and delete text
            for i in range(0, sample_rate):
                try:
                    # print(f'\nSampling round number {i+1} for {model.model_name}.')

                    cur_token_idx = token_idx.clone()
                    # We generate a random index from which we then take a sequence
                    random_idx = random.randint(16, len(cur_token_idx[0]) - sample_sequence_length)
                    sample_sequence_idx = cur_token_idx[:, random_idx: random_idx + sample_sequence_length]
                    sample_sequence = model.tokenizer.decode(sample_sequence_idx[0])
                    context_idx = cur_token_idx[:, :random_idx]
                    context = model.tokenizer.decode(context_idx[0])

                    # Now try to regenerate that sequence with the cur model
                    output = model.generate_k_with_probs(input_text=context, 
                                                         target_idx=sample_sequence_idx, 
                                                         max_length=sample_sequence_length, 
                                                         k=k,
                                                         temp=temp)
                    if(output == None):
                        return None
                    output['context'] = context
                    output['sample_sequence'] = sample_sequence
                    avg_prob = statistics.mean([s['target_prob'] for s in output['steps']])
                    # print(f"Predicted sequence: {output['generated_text']}")    
                    # print(f"Average target token probability {avg_prob}") 
                    model_result['sample_results'].append({
                        'avg_prob': avg_prob,
                        'model_outputs': output
                    })      
                except Exception as ex:
                    print(f'Caught an exception in one sample round. Model: {model.model_name};')
                    print(ex)
                    print(traceback.format_exc())

            model_result['avg_prob'] = statistics.mean(float(s['avg_prob']) for s in model_result['sample_results'])
            ensemble_results.append(model_result)
    
        metadata = {
            'full_text': text,
            'ensemble': [model.model_name for model in self.llm_ensemble],
            'sample_rate': sample_rate,
            'sample_sequence_length': sample_sequence_length,
            'min_token_length': self.min_token_length,
            'date': str(datetime.now()),
            'seed': seed,
            'signature': hashlib.sha256(text.encode('utf-8')).hexdigest()
        }

        # After we gather intel from our model ensemble, let our model decide over AI or not
        if self.luminar is not None:
            pred = self.luminar.predict(ensemble_results)
            metadata['is_AI'] = True if pred <= 0.4 else False
            metadata['pred'] = pred

        return {
            'metadata': metadata,
            'ensemble_results': ensemble_results
        }     

example_text_AI = '''
In his first State of the Union address, President Joe Biden delivered an optimistic outlook on the nation's economy, celebrating its remarkable rebound while drawing a clear contrast with the policies of his predecessor, Donald Trump. Against the backdrop of ongoing challenges posed by the pandemic and geopolitical tensions, Biden sought to instill confidence in the American people and reaffirm his administration's commitment to fostering inclusive growth and resilience.

Highlighting the progress made since taking office, President Biden touted the nation's robust economic recovery, citing significant job gains, declining unemployment rates, and steady GDP growth. He attributed these achievements to the successful implementation of his administration's economic agenda, including the passage of key legislation such as the American Rescue Plan and the Build Back Better Act.

In his address, Biden emphasized the importance of addressing longstanding economic disparities and rebuilding a more equitable and sustainable economy. He underscored his administration's efforts to invest in infrastructure, expand access to healthcare and education, and support working families through initiatives such as the child tax credit and affordable childcare.

Drawing a sharp distinction from the policies pursued by the Trump administration, President Biden criticized the previous administration's approach to economic governance, accusing it of favoring the wealthy and neglecting the needs of ordinary Americans. He pointed to the widening wealth gap, stagnant wages, and escalating inequality as evidence of the shortcomings of Trump's economic policies.

Biden also seized the opportunity to outline his vision for the future, emphasizing the importance of addressing pressing challenges such as climate change, healthcare reform, and racial injustice. He called for bipartisan cooperation and urged lawmakers to put aside partisan differences in pursuit of common goals that benefit all Americans.

As the nation grapples with the aftermath of the pandemic and seeks to chart a path forward, President Biden's State of the Union address served as a rallying cry for unity and resilience. While acknowledging the progress achieved, he acknowledged that much work remains to be done to build a more inclusive and prosperous future for all.'''

example_text_HU = '''
President Biden enters his State of the Union speech on Thursday with an economic record that has defied forecasters’ gloomy expectations, avoiding recession while delivering stronger growth and lower unemployment than predicted.

But polls suggest voters know relatively little about the legislation Mr. Biden has signed into law that seeks to boost the economy through spending and tax breaks for infrastructure, clean energy, semiconductors and more.

They remain frustrated over high prices, particularly for groceries and housing, even though the rapid inflation that defined Mr. Biden’s early years in office has cooled. Mr. Biden consistently trails his predecessor and likely November opponent, former President Donald J. Trump, on economic issues.

His speech on Thursday will try to make the case for the success of “Bidenomics.” Mr. Biden will begin to hint at what his agenda might bring in a second term, including efforts to increase corporate taxes and to reduce the cost of housing, one of the most tangible examples of what Mr. Biden calls his attempts to build an economy that prioritizes workers and the middle class.

Mr. Biden’s State of the Union speech will “discuss the historic achievements he’s delivered for the American people and lay out his vision for the future,” Lael Brainard, who heads the president’s National Economic Council, told reporters ahead of the speech. She stressed recent wage gains, low unemployment and new factory construction that she said were linked to Mr. Biden’s agenda.

Ms. Brainard and other administration officials said the president would try to draw sharp contrasts with Mr. Trump on economic issues during his annual speech, including on tax policy and reducing consumer costs. Mr. Biden’s aim is to cast Mr. Trump and his Republican Party as allies of the wealthy and large corporations instead of Americans who are struggling with rising costs.

Those contrasts will include policy departures from Mr. Trump’s legacy. Mr. Biden will propose raising the corporate income tax rate to 28 percent, up from the 21 percent rate that Mr. Trump signed into law in 2017. He will also call for increasing a new minimum tax on large corporations, which Mr. Biden signed into law in 2022, to 21 percent from 15 percent.

Mr. Biden will also propose ending the ability of corporations to deduct compensation costs for any employee who is paid more than $1 million per year.

The president’s allies in Washington diverge on what economic issues he should focus on in this week’s speech. But they roundly agree that he should claim credit for measures of economic strength on his watch, while promising to fight more to tame prices.
'''

if __name__ == '__main__':
    # Just for testing
    detector = Detector(models=['daryl149/llama-2-7b-chat-hf'])
    print(detector.detect(sample_sequence_length=32, text=example_text_HU)['metadata']['pred'])