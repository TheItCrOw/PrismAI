from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
from IPython.display import display, Markdown, Latex
import torch

class CausalLM():

    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, context, max_length=256):
        input_ids = self.tokenizer.encode(context, return_tensors='pt')
        # Check if the input sequence is longer than the model's maximum sequence length
        if input_ids.size(1) > self.model.config.max_position_embeddings:
            print("Input sequence is too long and will be truncated.")
            input_ids = input_ids[:, -(self.model.config.max_position_embeddings / 2)]

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
                              temp=0.9):

        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        # Check if the input sequence is longer than the model's maximum sequence length
        if input_ids.size(1) > self.model.config.max_position_embeddings:
            print("Input sequence is too long and will be truncated.")
            input_ids = input_ids[:, -self.model.config.max_position_embeddings]

        # TODO: current_ids is broken after truncation above if needed!
        current_ids = input_ids.clone()
        start_length = len(current_ids[0])
        steps = []

        for step in range(max_length):
            # Generate the next token
            with torch.no_grad():
                logits = self.model(current_ids).logits[:, -1, :] / temp
                
            # Apply softmax to obtain probabilities
            softmaxed_logits = torch.nn.functional.softmax(logits, dim=-1)

            # Use torch.multinomial to sample k tokens
            top_k_indices = torch.multinomial(softmaxed_logits, 5)
            top_k_tokens = self.tokenizer.decode(top_k_indices[0])

            # Extract probabilities for all sampled tokens
            top_k_probs = softmaxed_logits[0][top_k_indices[0]]

            # Extract the top 1 token and its probability
            top_1_prob = top_k_probs[top_k_probs.argmax()]
            top_1_index = top_k_indices[0][top_k_probs.argmax()]

            # If we have target tokens, we also want to know the prob of them
            # relative to the chosen token.
            target_prob = None
            if len(target_idx > 0) and len(target_idx -1 >= step):
                target_prob = softmaxed_logits[0][target_idx[step]]

            # Concatenate the top 1 token to current_ids
            current_ids = torch.cat([current_ids, top_1_index.reshape(1, -1)], dim=-1)

            steps.append({
                'step': step,
                'token_id': top_1_index,
                'token': self.tokenizer.decode(top_1_index),
                'token_prob': top_1_prob,
                'top_k_tokens': top_k_tokens,
                'top_k_token_ids': top_k_indices,
                'top_k_probs': top_k_probs,
                'target_prob': target_prob
            })

        return {
            'generated_text': self.tokenizer.decode(current_ids[0, start_length:], skip_special_tokens=True),
            'steps': steps
        }
