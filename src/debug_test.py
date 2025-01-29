import json

from bson import DBRef

from transition_scores.pre_processor.window import SlidingWindowTextPreProcessor
from transition_scores.scorer import OnnxTransitionScorer

pre_processor = SlidingWindowTextPreProcessor.from_pretrained(
    "gpt2",
    stride=32,
    max_length=128,
)
scorer = OnnxTransitionScorer(
    "/hot_storage/models/onnx/gpt2_onnx_o4/",
    batch_size=1,
    device="cuda",
    top_k=4,
)

dataset = [
    {
        "_ref_id": str(DBRef("lorem-ipsum", "paragraph-1")),
        "ref_id": str(DBRef("lorem-ipsum", "paragraph-1")),
        "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean a velit at nisl sagittis accumsan sit amet ut neque. Pellentesque a elit nec erat venenatis ultricies. Maecenas ac consequat velit, bibendum facilisis lorem. Duis est neque, molestie et pellentesque vitae, consectetur quis arcu. Vivamus eros neque, egestas non pretium ac, varius vel enim. Curabitur convallis vitae odio et egestas. Quisque rutrum augue vitae metus cursus consequat nec id elit. Mauris fringilla non justo et bibendum. ",
        # "chunks": [
        #     "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        #     "Aenean a velit at nisl sagittis accumsan sit amet ut neque.",
        #     "Pellentesque a elit nec erat venenatis ultricies.",
        #     "Maecenas ac consequat velit, bibendum facilisis lorem.",
        #     "Duis est neque, molestie et pellentesque vitae, consectetur quis arcu.",
        #     "Vivamus eros neque, egestas non pretium ac, varius vel enim.",
        #     "Curabitur convallis vitae odio et egestas.",
        #     "Quisque rutrum augue vitae metus cursus consequat nec id elit.",
        #     "Mauris fringilla non justo et bibendum.",
        # ],
    },
    {
        "_ref_id": str(DBRef("lorem-ipsum", "paragraph-2")),
        "ref_id": str(DBRef("lorem-ipsum", "paragraph-2")),
        "text": "Duis blandit, arcu nec facilisis molestie, elit tortor pretium neque, interdum tincidunt eros ligula eu diam. Aenean arcu massa, consequat quis sodales sed, vestibulum a dui. Pellentesque luctus aliquam dui a vehicula. Nunc commodo ante sed ante facilisis euismod. Donec ultrices ornare massa gravida pharetra. Sed iaculis velit in justo lacinia, quis rhoncus dolor tincidunt. Ut quam ligula, porttitor sit amet urna ac, aliquam efficitur mauris. Praesent et mollis felis. Suspendisse a facilisis mauris. Quisque enim tellus, varius sit amet interdum nec, blandit vel lectus. Suspendisse eget felis non enim vestibulum vehicula id eu odio. Nunc aliquet felis eget elit aliquam eleifend. Curabitur sed metus felis.",
        # "chunks": [
        #     "Duis blandit, arcu nec facilisis molestie, elit tortor pretium neque, interdum tincidunt eros ligula eu diam.",
        #     "Aenean arcu massa, consequat quis sodales sed, vestibulum a dui.",
        #     "Pellentesque luctus aliquam dui a vehicula.",
        #     "Nunc commodo ante sed ante facilisis euismod.",
        #     "Donec ultrices ornare massa gravida pharetra.",
        #     "Sed iaculis velit in justo lacinia, quis rhoncus dolor tincidunt.",
        #     "Ut quam ligula, porttitor sit amet urna ac, aliquam efficitur mauris.",
        #     "Praesent et mollis felis.",
        #     "Suspendisse a facilisis mauris.",
        #     "Quisque enim tellus, varius sit amet interdum nec, blandit vel lectus.",
        #     "Suspendisse eget felis non enim vestibulum vehicula id eu odio.",
        #     "Nunc aliquet felis eget elit aliquam eleifend.",
        #     "Curabitur sed metus felis. ",
        # ],
    },
]

print(json.dumps(scorer.process(dataset, pre_processor), indent=2))
