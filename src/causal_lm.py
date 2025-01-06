import numpy as np
import torch
import spacy
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
from IPython.display import display, Markdown, Latex
from decoding_strategy import DecodingStrategy

class CausalLM():

    def __init__(self, model_name, include_spacy=True):
        self.model_name = model_name
        self.nlp = None
        if(include_spacy):
            self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                       trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          trust_remote_code=True)
        # Automatically select the GPU with the most available memory
        if torch.cuda.is_available():
            free_gpus = [
                (i, torch.cuda.mem_get_info(i)[0])
                for i in range(torch.cuda.device_count())
            ]
            best_gpu = max(free_gpus, key=lambda x: x[1])[0]
            self.device = torch.device(f"cuda:{best_gpu}")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device).eval()
        self.set_seed(42)
        print(f'Created model {model_name} to device {self.device}')

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def generate(self, context, max_length=256):
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        max_input_length = self.model.config.max_position_embeddings - max_length
        
        # Truncate the input if it exceeds the maximum length allowed for generation
        if input_ids.size(1) > max_input_length:
            print(f"Input sequence is too long. Truncating to fit within {max_input_length} tokens.")
            input_ids = input_ids[:, :max_input_length]

        attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
        output = self.model.generate(
            input_ids, 
            max_length=max_length,
            attention_mask=attention_mask
        )

        # Extract and return only the newly generated tokens
        generated_tokens = output[0, input_ids.size(1):]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text
    
    def generate_k_with_probs(self, 
                              input_text,
                              target_idx=[], 
                              k=5, 
                              max_length=32, 
                              temp=0.9,
                              p=0.15,
                              beam_width=5,
                              decoding_strategy='top_k'):

        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        max_input_length = self.model.config.max_position_embeddings - max_length
        
        # Truncate the input if it exceeds the maximum length allowed for generation
        if input_ids.size(1) > max_input_length:
            #print(f"Input sequence is too long. Truncating to fit within {max_input_length} tokens.")
            input_ids = input_ids[:, :max_input_length]

        current_ids = input_ids.clone()
        start_length = len(current_ids[0])
        steps = []
        decoder_strategy = DecodingStrategy(tokenizer=self.tokenizer, temp=temp, k=k, p=p)
        beam_search_sequences = [([], 1.0)] 

        for step in range(max_length):
            context = self.tokenizer.decode(current_ids[0])

            with torch.no_grad():
                logits = self.model(current_ids.to(self.device)).logits[:, -1, :] / temp

            # Beam Search is fundamentally different from other decoding strategies, so we handle it exlusively
            if decoding_strategy == 'beam_search':
                top_k_indices, top_k_tokens, top_k_probs, top_1_prob, top_1_index, beam_search_sequences = decoder_strategy.beam_search(
                    logits=logits, 
                    beam_width=beam_width, 
                    current_sequences=beam_search_sequences
                )
                if target_idx == None:
                    current_ids = torch.cat([current_ids.cpu(), torch.tensor([seq[-1] for seq, _ in beam_search_sequences]).unsqueeze(0).cpu()], dim=-1)
            else:
                softmaxed_logits, top_k_indices, top_k_tokens, top_k_probs, top_1_prob, top_1_index = decoder_strategy.apply_strategy(
                    strategy=decoding_strategy, 
                    logits=logits
                )
                if target_idx == None:
                    current_ids = torch.cat([current_ids.cpu(), top_1_index.reshape(1, -1).cpu()], dim=-1)

            # If we have specific target idx of tokens, then use those to continue the sequence
            # if not, then as written as above, use the tokens we've just created.
            if target_idx != None:
                current_ids = torch.cat([current_ids.cpu(), target_idx[:, step].unsqueeze(dim=1).cpu()], dim=-1)

            target = None
            target_prob = None
            target_pos = "UNKOWN"
            if target_idx is not None and len(target_idx) > 0 and len(target_idx[0]) - 1 >= step:
                target = self.tokenizer.decode(target_idx[0][step]).strip()
                target_prob = softmaxed_logits[0][target_idx[0][step]]
                if self.nlp is not None:
                    doc = self.nlp(target)
                    if len(doc) > 0:
                        target_pos = doc[0].pos_

            steps.append({
                'step': step,
                'context': context,
                'token_id': top_1_index.item(),
                'token': self.tokenizer.decode(top_1_index),
                'token_prob': top_1_prob.item(),
                'top_k_tokens': top_k_tokens,
                'top_k_token_ids': top_k_indices.tolist()[0],
                'top_k_probs': top_k_probs.tolist() if decoding_strategy != 'beam_search' else top_k_probs, # type: ignore
                'target_prob': target_prob.item() if target_prob is not None else "",
                'target': target,
                'target_pos': target_pos
            })

        return {
            'generated_text': self.tokenizer.decode(current_ids[0, start_length:], skip_special_tokens=True),
            'steps': steps
        }

