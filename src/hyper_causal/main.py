import argparse
from flask import (
    Flask,
    g,
    render_template,
    request,
    jsonify,
    current_app,
    redirect,
    url_for,
)
import sys
import os
import torch
import traceback
from hyper_causal_dto import HyperCausal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from causal_lm import CausalLM

app = Flask(
    __name__,
    template_folder="./web/templates",
    static_folder="./web/static",
    static_url_path="/static",
)


@app.route("/")
def index():
    app_mode = get_app_mode()
    print("App mode: " + app_mode)

    # Dependant on the app mode, we return different views as the index.
    if app_mode == "CLI":
        return redirect(url_for("get_hyper_causal", id="CLI"))
    elif app_mode == "WEB":
        return render_template(
            "html/index.html", instances=get_all_hyper_causal_instances()
        )
    else:
        return "Encountered unknown app mode - it must be either 'CLI' or 'WEB'. You should restart the service."


@app.route("/hyper_causal")
def get_hyper_causal():
    try:
        id = str(request.args.get("id"))
        hyper_causal = get_hyper_causal_instance(id)
        return get_hyper_causal_view(hyper_causal)
    except Exception as ex:
        print("Error getting a HyperCausal view:")
        print(ex)
        print(traceback.format_exc())
        return "Error: Couldn't get the HyperCausal view: " + str(ex)


@app.route("/api/hyper_causal/new", methods=["POST"])
def post_new_hyper_causal_instance():
    result = {"status": 400, "message": ""}
    try:
        if request.is_json:
            print(request.get_json())
            hyper_causal = HyperCausal.from_request(request)
            # Empty spaces at the start and end makes it harder for the llm to predict.
            hyper_causal.input = hyper_causal.input.strip()
            cache_hyper_causal_instance(hyper_causal)
            # Create the LLM for the hypercausal if its not cached already.
            get_llm(hyper_causal.model_name)
            result["status"] = 200
            result["id"] = hyper_causal.id
    except Exception as ex:
        result["message"] = str(ex)
        print("Couldn't POST HyperCausal instance and create the Causal LLM: ")
        print(ex)
        print(traceback.format_exc())
    return jsonify(result)


@app.route("/api/tokens/next", methods=["POST"])
def get_next_tokens():
    result = {
        "status": 400,
    }
    try:
        if request.is_json:
            hyper_causal = HyperCausal.from_request(request, False)
            llm_instance = get_llm(hyper_causal.model_name)

            next = llm_instance.generate_k_with_probs(
                input_text=hyper_causal.input,
                target_idx=None,
                k=hyper_causal.k,
                temp=hyper_causal.temp,
                p=hyper_causal.p,
                beam_width=hyper_causal.beam_width,
                decoding_strategy=hyper_causal.decoding_strategy,
                max_length=1,
            )
            # print(next)

            result["result"] = next  # type: ignore
            result["status"] = 200
    except Exception as ex:
        print("Couldn't generate next token branches: ")
        print(ex)
        print(traceback.format_exc())
    return jsonify(result)


def get_hyper_causal_view(hyper_causal_instance):
    return render_template("html/hyper_causal.html", hyper_causal=hyper_causal_instance)


def parse_arguments():
    parser = argparse.ArgumentParser(description="HyperCausal CLI arguments")
    # Possible app_modes are CLI or WEB
    parser.add_argument(
        "--app_mode",
        type=str,
        default="WEB",
        help='Default "CLI": Enter the prompt and parameters directly through the cli.\n "WEB": Enter the prompt and parameters through a web interface.',
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP address")
    parser.add_argument("--port", type=int, default=5678, help="Port number")
    parser.add_argument(
        "--debug",
        type=bool,
        default=True,
        help="True/False whether the app should be started in debug mode (required True for hot reload)",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="GPT2",
        help="The Causal LLM which will be used to generate the tokens and probabilities. Please visit https://huggingface.co/docs/transformers/tasks/language_modeling for a detailled list of avaiablabe models. Fine-tuned versions are also possible.",
    )
    parser.add_argument(
        "--k", type=int, default=3, help="How many alternative branches to visualize."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=7,
        help="The max amount of tokens to generate.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Once upon a time there was a boy",
        help="The input text which the model will generate from.",
    )
    parser.add_argument(
        "--tree_style",
        type=str,
        default="breadth-first",
        help='The style which the tree is being built with, currently either "breadth-first" or "depth-first".',
    )
    parser.add_argument(
        "--decoding_strategy",
        type=str,
        default="top_k",
        help="Decoding strategy of text generation, see e.g.: https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html. Possible decodings: top_k, greedy_top_k, top_p, beam_search.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default="0.15",
        help="Only relevant for top_p decoding. Probability cutoff threshold for selected tokens, and hence ranges from 0 to 1.",
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default="5",
        help="Only relevant for beam_search decoding. Sets the number of beams, min 1 and max 20.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default="0.9",
        help='Sets the temperature of the causal LLM, which stands for the randomness, sometimes referred to as "creativeness" of the model.',
    )
    return parser.parse_args()


def create_llm(model_name: str) -> CausalLM:
    print(
        "Create "
        + model_name
        + "to device: "
        + str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    )
    llm_instance = CausalLM(model_name=model_name, include_spacy=False)
    return llm_instance


def get_llm(model_name: str) -> CausalLM:
    if model_name not in current_app.config:
        current_app.config[model_name] = create_llm(model_name)
    return current_app.config[model_name]


def cache_hyper_causal_instance(hyper_causal: HyperCausal):
    if hyper_causal.id not in current_app.config:
        current_app.config["instance_" + hyper_causal.id] = hyper_causal


def get_hyper_causal_instance(hyper_causal_id: str) -> HyperCausal:
    return current_app.config["instance_" + hyper_causal_id]


def get_all_hyper_causal_instances():
    return [
        value
        for key, value in current_app.config.items()
        if key.startswith("instance_")
    ]


def get_app_mode() -> str:
    if "app_mode" not in current_app.config:
        args = parse_arguments()
        current_app.config["app_mode"] = args.app_mode
    return current_app.config["app_mode"]


def main():
    args = parse_arguments()
    host = args.host
    port = args.port
    print("Starting HyperCausal in mode: " + args.app_mode)

    # In CLI mode, we create a single hyper_causal instance from the parameters and that's it.
    if args.app_mode == "CLI":
        with app.app_context():
            hyper_causal = HyperCausal(
                args.input,
                args.llm,
                args.k,
                args.max_tokens,
                args.temp,
                args.p,
                args.beam_width,
                args.decoding_strategy,
                args.tree_style,
            )
            hyper_causal.id = "CLI"
            cache_hyper_causal_instance(hyper_causal)
            print("Since app_mode = CLI, HyperCausal instance was created.")

            # Also, create the LLM upon start already
            current_app.config[args.llm] = create_llm(args.llm)
            print(f"Since app_mode = CLI, LLM {args.llm} was created.")

    app.run(host=host, port=port, debug=args.debug)


if __name__ == "__main__":
    main()
