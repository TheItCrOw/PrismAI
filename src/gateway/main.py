import sys
import os
import traceback

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
    return render_template(
        "html/index.html", instances=['test', 'test2']
    )

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

            detector = get_detector("")
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
        # TODO: Hardcoded Model path for now!
        # /storage/projects/boenisch/PrismAI/models/luminar_sequence/de___en/trvaaa3c
        # tiiuae/falcon-7b
        # /storage/projects/boenisch/PrismAI/models/luminar_sequence/PrismAI_v2-encoded-gpt2/fa05u0tn
        current_app.config[model_name] = LuminarSequenceDetector(model_path="/storage/projects/boenisch/PrismAI/models/luminar_sequence/PrismAI_v2-encoded-gpt2/fa05u0tn", 
                                                                 feature_agent="gpt2", 
                                                                 device=get_best_device())
    return current_app.config[model_name]

def main():
    # TODO: make this configurable
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    main()