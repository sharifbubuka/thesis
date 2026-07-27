import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")

from src.utils.visualization import plot_patch_heatmap


def test_patch_heatmap_displays_question_and_answer_below_title() -> None:
    figure = plot_patch_heatmap(
        Image.new("RGB", (32, 32)),
        np.ones(4),
        question="What is shown in this image?",
        predicted_answer="A cat",
    )

    image_axis = figure.axes[0]
    assert image_axis.get_title() == "Image-patch contribution"
    assert [text.get_text() for text in image_axis.texts] == [
        "Question: What is shown in this image?\nPredicted answer: A cat"
    ]
    assert image_axis.texts[0].get_position()[1] > 1

    plt.close(figure)
