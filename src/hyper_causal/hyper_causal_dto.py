import uuid

class HyperCausal:
    '''
    We have some state properties which we bundle into a struct to hold a HyperCausal Instance state.
    '''

    @staticmethod
    def from_request(request):
        data = request.get_json()
        
        # Define expected properties and their types
        expected_fields = {
            'input': str,
            'llm': str,
            'k': int,
            'maxTokens': int,
            'temp': float,
            'p': float,
            'beamWidth': int,
            'decodingStrategy': str,
            'treeStyle': str
        }
        
        # Validate and extract data
        validated_data = {}
        for field, field_type in expected_fields.items():
            if field not in data:
                raise ValueError(f"Missing required property: {field}")
            try:
                validated_data[field] = field_type(data[field])
                if(field_type == str and data[field] == ''):
                    raise AssertionError(f"The string field '{field}' was empty.")
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid type for property '{field}': expected {field_type.__name__}, got {type(data[field]).__name__}.")
        
        # Create and return the HyperCausal object using validated data
        return HyperCausal(
            input=validated_data['input'],
            model_name=validated_data['llm'],
            k=validated_data['k'],
            max_tokens=validated_data['maxTokens'],
            temp=validated_data['temp'],
            p=validated_data['p'],
            beam_width=validated_data['beamWidth'],
            decoding_strategy=validated_data['decodingStrategy'],
            tree_style=validated_data['treeStyle']
        )

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