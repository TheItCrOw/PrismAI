import sys
import os
import traceback
import argparse

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    current_app
)

# to get Luminar models and more
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from luminar.detector import LuminarSequenceDetector
from luminar.utils import get_best_device
from luminar.utils.visualization import visualize_detection

# Create the flask application
app = Flask(
    __name__,
    template_folder="./web/templates",
    static_folder="./web/static",
    static_url_path="/static",
)


@app.route("/")
def index():
    return render_template("html/index.html", instances=['test', 'test2'])


@app.route("/api/detect", methods=["POST"])
def post_new_detection():
    response = {"status": 404, "payload": ""}
    print("New detection request.")
    try:
        if not request.is_json:
            response["payload"] = "The request needs to be in JSON format."
        else:
            data = request.get_json()
            print("Payload: ", data)
            document = data["document"]

            detector = get_detector("default_model")
            result = detector.detect(document)
            html_output = visualize_detection(document, result)
            result["visualization"] = html_output
            response["status"] = 200
            response["payload"] = result
            print("Returning: ", result)
    except Exception as ex:
        response["payload"] = "There was an unknown error while trying to detect."
        print(ex)
        print(traceback.format_exc())

    return jsonify(response)


def get_detector(model_name: str) -> LuminarSequenceDetector:
    if model_name not in current_app.config:
        current_app.config[model_name] = LuminarSequenceDetector(
            model_path=current_app.config["model_path"],
            feature_agent=current_app.config["feature_agent"],
            device=get_best_device()
        )
    return current_app.config[model_name]


def main():
    parser = argparse.ArgumentParser(description="Run Luminar Flask API")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/storage/projects/boenisch/PrismAI/models/luminar_sequence/PrismAI_v2-encoded-gpt2/e1s2k2du",
        help="Path to the LuminarSequenceDetector model (default: old hardcoded path)"
    )
    parser.add_argument(
        "--feature-agent",
        type=str,
        default="gpt2",
        help="Feature agent to use (default: gpt2)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the Flask app (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the Flask app (default: 8080)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Enable debug mode"
    )

    args = parser.parse_args()
    app.config["model_path"] = args.model_path
    app.config["feature_agent"] = args.feature_agent
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
