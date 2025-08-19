import matplotlib.pyplot as plt
import seaborn as sns
import html

from matplotlib.image import AxesImage
from numpy.typing import NDArray

def visualize_features(features: NDArray) -> AxesImage:
    fig = plt.imshow(
        features,
        cmap=sns.cubehelix_palette(as_cmap=True),
        vmin=min(0.0, features.min()),
        vmax=max(1.0, features.max()),
    )
    fig.axes.set_axis_off()
    plt.tight_layout()
    return fig


def visualize_detection(document: str, detection_result: dict, threshold: float = 0.0) -> str:
    """
    Visualizes AI-generated span predictions inline in HTML using color highlighting.
    :param document: Original document text.
    :param detection_result: Output from `detect_sequences()`.
    :param threshold: Minimum probability to highlight a span.
    :return: Rendered HTML string for use in notebooks and elsewhere.
    """
    char_spans = detection_result["char_spans"]
    probs = detection_result["probs"]

    # Sort by start index to render in order
    sorted_spans = sorted(zip(char_spans, probs), key=lambda x: x[0][0])

    def get_color(prob: float) -> str:
        red = int(255 * prob)
        green = int(255 * (1 - prob))
        return f"rgba({red}, {green}, 100, 0.5)"  # soft gradient

    final_html = ""
    last_idx = 0

    for (start, end), prob in sorted_spans:
        if start >= end or start >= len(document):
            continue

        end = min(end, len(document))

        # Add text before the span
        if last_idx < start:
            final_html += html.escape(document[last_idx:start])

        # Highlight span if above threshold
        span_text = html.escape(document[start:end])
        if prob >= threshold:
            color = get_color(prob)
            final_html += (
                f'<span style="background-color:{color}; padding:1px 3px; border-radius:3px; margin-right:2px;" '
                f'title="AI probability: {prob:.2f}">{span_text}</span>'
                f'<span style="font-size: 0.8em; color: #555; margin-right: 6px; background-color: ghostwhite; border: gray 1px solid; padding:1px 3px; border-radius:3px;">'
                f'({prob:.2f}%)'
                f'</span>'
            )
        else:
            final_html += span_text

        last_idx = end

    # Add remaining text
    if last_idx < len(document):
        final_html += html.escape(document[last_idx:])

    return f"<div style='line-height:1.6; font-family:monospace;'>{final_html}</div>"
