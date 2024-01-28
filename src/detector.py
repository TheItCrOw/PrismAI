from causal_lm import CausalLM
import random
import statistics

class Detector():
    '''A class that tries to identify AI texts'''

    def __init__(self, models=['gpt2']):
        self.llm_ensemble = [CausalLM(m) for m in models]
        self.min_token_length = 32
        print(f'Model ensemble: {[x.model_name for x in self.llm_ensemble]}')

    def detect(self, 
               text, 
               sample_rate=6,
               sample_sequence_length=12):
        '''A function that tries to detect AI sequences in a text'''

        ensemble_avg = []
        # We have an ensemble of models which we use to detect
        for model in self.llm_ensemble:
            print(f'Using {model.model_name} ===')
            model_avg = []

            # Tokenize the text
            token_idx = model.tokenizer.encode(text, return_tensors='pt')
            # I dont think we can handle very short texts. We will see.
            if(len(token_idx[0]) < self.min_token_length):
                raise Exception(f'The text must be at least 32 tokens long, got only {len(token_idx[0])}.')

            # We will randomly sample and delete text
            for i in range(0, sample_rate):
                try:
                    print(f'\nSampling round number {i+1} for {model.model_name}.')

                    cur_token_idx = token_idx.clone()
                    # We generate a random index from which we then take a sequence
                    random_idx = random.randint(16, len(cur_token_idx[0]) - sample_sequence_length)
                    sample_sequence_idx = cur_token_idx[:, random_idx: random_idx + sample_sequence_length]
                    context_idx = cur_token_idx[:, :random_idx]
                    context = model.tokenizer.decode(context_idx[0])
                    print(f'Context sample: {context}')
                    print(f'Sampling sequence: {model.tokenizer.decode(sample_sequence_idx[0])}')

                    # Now try to regenerate that sequence with the cur model
                    output = model.generate_k_with_probs(context, sample_sequence_idx[0], max_length=sample_sequence_length)
                    avg_prob = statistics.mean([s['target_prob'].item() for s in output['steps']])
                    print(f"Predicted sequence: {output['generated_text']}")    
                    print(f"Average target token probability {avg_prob}") 
                    model_avg.append(avg_prob)      
                except Exception as ex:
                    print(f'Caught an exception in one sample round. Model: {model.model_name};')
                    print(ex)

            ensemble_avg.append(statistics.mean(model_avg))

        return statistics.mean(ensemble_avg)     
            

test_text = '''
Do you have a problem that you are struggling to solve? Why don't you ask your friends for advice? When people ask for advice on solving a problem, often times they speak to more than one person. This is because different views are better for figuring out a tough problem, many opinions are better than one, and other people may have experienced a problem like yours and may be able to help you in making better decisions.
'''


if __name__ == '__main__':
    # Just for testing
    detector = Detector()
    detector.detect(test_text)