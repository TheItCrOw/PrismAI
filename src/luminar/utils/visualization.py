import matplotlib.pyplot as plt
import seaborn as sns
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
