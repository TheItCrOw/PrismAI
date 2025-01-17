import torch
import torch.nn.functional as F

class DecodingStrategy():
    '''
    See also: https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html
    '''

    def __init__(self, tokenizer, temp, k, p):
        self.temp = temp
        self.k = k
        self.p = p
        self.tokenizer = tokenizer

    def apply_strategy(self, strategy, logits):
        if(strategy == 'top_k'):
            return self.top_k(logits)
        elif(strategy == 'greedy_top_k'):
            return self.greedy_top_k(logits)
        elif(strategy == 'top_p'):
            return self.top_p(logits)
        else:
            raise ValueError('Unknown strategy for decoding: ' + strategy)

    def beam_search(self, logits, beam_width, current_sequences):
        '''
        Beam Search Decoding
        '''
        softmaxed_logits = F.softmax(logits, dim=-1)
        probs = softmaxed_logits[0]
        top_k_probs, top_k_indices = torch.topk(probs, beam_width)

        if current_sequences is None:
            current_sequences = [([], 1.0)]

        all_candidates = []

        for sequence, seq_prob in current_sequences:
            for i in range(beam_width):
                new_sequence = sequence + [top_k_indices[i].item()]
                new_prob = seq_prob * top_k_probs[i].item()
                all_candidates.append((new_sequence, new_prob))

        all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        best_candidates = all_candidates[:beam_width]

        best_sequence = best_candidates[0][0]
        best_prob = best_candidates[0][1]
        top_k_tokens = [self.tokenizer.decode(idx) for idx in top_k_indices]
        top_k_probs = top_k_probs.tolist()
        top_1_index = torch.tensor([best_sequence[-1]])
        top_1_prob = torch.tensor([best_prob])

        return top_k_indices, top_k_tokens, top_k_probs, top_1_prob, top_1_index, best_candidates

    def top_p(self, logits):
        '''top_p decoding, also known as "Nucleus Sampling"'''
        softmaxed_logits = torch.nn.functional.softmax(logits, dim=-1)
        probs = softmaxed_logits[0]
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Filter tokens based on cumulative probability threshold p
        top_k_mask = cumulative_probs <= self.p
        top_k_indices = sorted_indices[top_k_mask]
        top_k_probs = sorted_probs[top_k_mask]

        # If no tokens meet the threshold, use k tokens
        if len(top_k_indices) == 0:
            print('No p tokens found, taking k')
            top_k_indices = sorted_indices[:self.k]
            top_k_probs = sorted_probs[:self.k]

        # Sample from filtered tokens
        sampled_index = torch.multinomial(top_k_probs, 1).item()
        top_1_index = top_k_indices[sampled_index] # type: ignore
        top_1_prob = top_k_probs[sampled_index] # type: ignore
        top_k_tokens = [self.tokenizer.decode(idx) for idx in top_k_indices]
        return softmaxed_logits, top_k_indices, top_k_tokens, top_k_probs, top_1_prob, top_1_index

    def greedy_top_k(self, logits):
        '''
        Greedy decoding, meaning we take the top k tokens.
        '''
        softmaxed_logits = torch.nn.functional.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(softmaxed_logits[0], self.k)
        top_k_tokens = [self.tokenizer.decode(idx) for idx in top_k_indices]
        top_1_index = torch.tensor(top_k_indices[0].item())
        top_1_prob = torch.tensor(top_k_probs[0].item())
        return softmaxed_logits, top_k_indices, top_k_tokens, top_k_probs, top_1_prob, top_1_index
    
    def top_k(self, logits):
        '''Standard top k decoding with multinomial sampling'''
        softmaxed_logits = torch.nn.functional.softmax(logits, dim=-1)
        top_k_indices = torch.multinomial(softmaxed_logits, min(self.k, len(softmaxed_logits[0])))
        top_k_tokens = [self.tokenizer.decode(t ,skip_special_tokens=False).strip() for t in top_k_indices[0]]
        top_k_probs = softmaxed_logits[0][top_k_indices[0]]
        top_1_prob = top_k_probs[top_k_probs.argmax()]
        top_1_index = top_k_indices[0][top_k_probs.argmax()]
        return softmaxed_logits, top_k_indices, top_k_tokens, top_k_probs, top_1_prob, top_1_index
    

