import argparse
from flask import Flask, g, render_template
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from causal_lm import CausalLM

app = Flask(__name__, template_folder="./web/templates", static_folder="./web/static", static_url_path='/static')

app.config['llm'] = None

@app.route('/')
def hello_world():
    # llm_instance = get_llm()
    return render_template("html/index.html", llm="LLM")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Flask App with Command Line Arguments')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host IP address')
    parser.add_argument('--port', type=int, default=5678, help='Port number')
    parser.add_argument('--debug', type=bool, default=True, help='True/False whether the app should be started in debug mode (required True for hot reload)')
    parser.add_argument('--llm', type=str, default="GPT2", help='llm string to store globally and pass to CausalLM')
    return parser.parse_args()

def create_llm(llm_arg):
    llm_instance = CausalLM(llm_arg, include_spacy=False)
    return llm_instance

def get_llm():
    if 'llm' not in g:
        args = parse_arguments()
        g.llm = create_llm(args.llm)
    return g.llm

def main():
    args = parse_arguments()
    host = args.host
    port = args.port
    app.run(host=host, port=port, debug=args.debug)

if __name__ == '__main__':
    main()