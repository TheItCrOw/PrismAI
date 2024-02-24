import argparse
from flask import Flask, g, render_template, request, jsonify, current_app
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from causal_lm import CausalLM

app = Flask(__name__, template_folder="./web/templates", static_folder="./web/static", static_url_path='/static')

@app.route('/')
def hello_world():
    llm_instance = get_llm()
    return render_template("html/index.html", 
                           llm=llm_instance.model_name, 
                           input='Steve Jobs was a',
                           max_new_tokens=7,
                           k=get_k())

@app.route('/api/tokens/next', methods=['POST'])
def get_next_tokens():
    result = {
        'status': 400,
    }
    try:
        if request.is_json:
            data = request.get_json()
            input = str(data.get('input'))            

            llm_instance = get_llm()
            next = llm_instance.generate_k_with_probs(
                input_text=input,
                target_idx=None,
                k = get_k(),
                max_length=1)

            result['result'] = next # type: ignore
            result['status'] = 200
            print(result)
    except Exception as ex:
        print("Couldn't generate next token branches: ")
        print(ex)
    return jsonify(result)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Flask App with Command Line Arguments')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host IP address')
    parser.add_argument('--port', type=int, default=5678, help='Port number')
    parser.add_argument('--debug', type=bool, default=True, help='True/False whether the app should be started in debug mode (required True for hot reload)')
    parser.add_argument('--llm', type=str, default="GPT2", help='llm string to store globally and pass to CausalLM')
    parser.add_argument('--k', type=int, default=3, help='How many alternative branches to visualize.')
    return parser.parse_args()

def create_llm(llm_arg):
    llm_instance = CausalLM(llm_arg, include_spacy=False)
    return llm_instance

def get_k():
    if 'k' not in current_app.config:
        args = parse_arguments()
        current_app.config['k'] = args.k
    return current_app.config['k']

def get_llm():
    if 'llm' not in current_app.config:
        args = parse_arguments()
        current_app.config['llm'] = create_llm(args.llm)
    return current_app.config['llm']

def main():
    args = parse_arguments()
    host = args.host
    port = args.port
    app.run(host=host, port=port, debug=args.debug)

if __name__ == '__main__':
    main()