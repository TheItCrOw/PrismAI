import torch
import numpy as np

class Luminar:
    '''
    The Luminar model trained to detect AI generated text, trained on CNN.
    '''

    def __init__(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval() 
        pass

    def predict(self, ensemble_output):
        '''
        Predicts whether the given ensemble output if from an AI text or Human text
        AI: 0 -> 0.3
        HU: > 0.3 probably
        '''
        with torch.no_grad():
            pred_tensor = self.model(self.ensemble_output_to_input(ensemble_output))
            return torch.mean(pred_tensor).item()
        
    def ensemble_output_to_input(self, output):
        '''
        Takes the outputs from the causal model ensemble and transforms them into matrices
        for the model to take as an input.
        '''
        results = output[0]['sample_results']
        rows = []

        for result in results:
            steps = result['model_outputs']['steps']
            steps.sort(key= lambda x: x['step'])
            columns = []
            for step in steps:
                prob = step['target_prob']
                # Common target stepwords, we mask out
                #if step['target'].lower() in english_stopwords:
                #    prob = 0
                channels = [prob]
                # Again, mask out common stopwords
                top_k_tokens = step['top_k_tokens']
                top_k_probs = step['top_k_probs']
                for i in range(0 , len(top_k_tokens)):
                    token = top_k_tokens[i]
                    #if token.lower() in english_stopwords:
                    #    top_k_probs[i] = 0
                top_k_probs.sort()
                channels = channels + top_k_probs
                columns.append(channels)
                
            rows.append(columns)

        tensor = torch.tensor(rows, dtype=torch.float32).permute(2, 0, 1)
        return tensor
        