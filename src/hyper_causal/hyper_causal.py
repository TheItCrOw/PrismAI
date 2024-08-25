import uuid

class HyperCausal:
    '''
    We have some state properties which we bundle into a struct to hold a HyperCausal Instance state.
    '''

    def __init__(self, input, model_name, k, max_tokens, temp, p, beam_width, decoding_strategy, tree_style):
        self.id = str(uuid.uuid4())
        self.model_name = model_name
        self.input = input
        self.k = k
        self.max_tokens = max_tokens
        self.tree_style = tree_style
        self.temp = temp
        self.p = p
        self.beam_width = beam_width
        self.decoding_strategy = decoding_strategy