import argparse
from flask import Flask, g, render_template, request, jsonify, current_app
import sys
import os
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from causal_lm import CausalLM

app = Flask(__name__, template_folder="./web/templates", static_folder="./web/static", static_url_path='/static')

@app.route('/')
def index():
    llm_instance = get_llm()
    app_mode = get_app_mode()
    print('App mode: ' + app_mode)
    # Dependant on the app mode, we return different templates
    if(app_mode == 'CLI'):
        return render_template("html/hyper_causal.html", 
                            llm=llm_instance.model_name, 
                            input=get_input(),
                            max_new_tokens=get_max_tokens(),
                            k=get_k(),
                            tree_style=get_tree_style())
    elif(app_mode == 'WEB'):
        return render_template("html/index.html")
    else: 
        return "Encountered unknown app mode - it must be either 'CLI' or 'WEB'."

@app.route('/api/tokens/next', methods=['POST'])
def get_next_tokens():
    result = {
        'status': 400,
    }
    try:
        if request.is_json:
            data = request.get_json()
            input = str(data.get('input'))
            overwrite_k = int(data.get('overwriteK'))            

            llm_instance = get_llm()
            next = llm_instance.generate_k_with_probs(
                input_text=input,
                target_idx=None,
                k = get_k() if overwrite_k == -1 else overwrite_k,
                max_length=1)
            # print(next)

            result['result'] = next # type: ignore
            result['status'] = 200
    except Exception as ex:
        print("Couldn't generate next token branches: ")
        print(ex)
        print(traceback.format_exc())
    return jsonify(result)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Flask App with Command Line Arguments')
    # Possible app_modes are CLI or WEB
    parser.add_argument('--app_mode', type=str, default='WEB', help='Default "CLI": Enter the prompt and parameters directly through the cli.\n "WEB": Enter the prompt and parameters through a web interface.')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host IP address')
    parser.add_argument('--port', type=int, default=5678, help='Port number')
    parser.add_argument('--debug', type=bool, default=True, help='True/False whether the app should be started in debug mode (required True for hot reload)')
    parser.add_argument('--llm', type=str, default="GPT2", help='The Causal LLM which will be used to generate the tokens and probabilities. Please visit https://huggingface.co/docs/transformers/tasks/language_modeling for a detailled list of avaiablabe models. Fine-tuned versions are also possible.')
    parser.add_argument('--k', type=int, default=3, help='How many alternative branches to visualize.')
    parser.add_argument('--max_tokens', type=int, default=7, help='The max amount of tokens to generate.')
    parser.add_argument('--input', type=str, default="Once upon a time there was a boy", help='The input text which the model will generate from.')
    parser.add_argument('--tree-style', type=str, default="breadth-first", help='The style which the tree is being built with, currently either "breadth-first" or "depth-first".')
    return parser.parse_args()

def create_llm(llm_arg):
    llm_instance = CausalLM(llm_arg, include_spacy=False)
    return llm_instance

def get_k():
    if 'k' not in current_app.config:
        args = parse_arguments()
        current_app.config['k'] = args.k
    return current_app.config['k']

def get_max_tokens():
    if 'max_tokens' not in current_app.config:
        args = parse_arguments()
        current_app.config['max_tokens'] = args.max_tokens
    return current_app.config['max_tokens']

def get_input():
    if 'input' not in current_app.config:
        args = parse_arguments()
        current_app.config['input'] = args.input
    return current_app.config['input']

def get_llm():
    if 'llm' not in current_app.config:
        args = parse_arguments()
        current_app.config['llm'] = create_llm(args.llm)
    return current_app.config['llm']

def get_app_mode():
    if 'app_mode' not in current_app.config:
        args = parse_arguments()
        current_app.config['app_mode'] = args.app_mode
    return current_app.config['app_mode']

def get_tree_style():
    if 'tree_style' not in current_app.config:
        args = parse_arguments()
        current_app.config['tree_style'] = args.tree_style
    return current_app.config['tree_style']

def main():
    args = parse_arguments()
    host = args.host
    port = args.port
    app.run(host=host, port=port, debug=args.debug)

if __name__ == '__main__':
    main()