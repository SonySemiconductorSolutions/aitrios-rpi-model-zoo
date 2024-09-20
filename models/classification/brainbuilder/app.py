import argparse
import time
from typing import List

import cv2
import numpy as np
from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500

last_detections = []
LABELS = None


class Classification:
    def __init__(self, idx: int, score: float):
        """Create a Classification object, recording the idx and score."""
        self.idx = idx
        self.score = score


def get_label(request: CompletedRequest, idx: int) -> str:
    """Retrieve the label corresponding to the classification index."""
    global LABELS
    if LABELS is None:
        with open(args.labels, "r") as f:
            LABELS = f.read().splitlines()
    return LABELS[idx]


def parse_and_draw_classification_results(request: CompletedRequest):
    """Analyse and draw the classification results in the output tensor."""
    results = parse_classification_results(request)
    draw_classification_results(request, results)


def parse_classification_results(request: CompletedRequest) -> List[Classification]:
    """Parse the output tensor into the classification results above the threshold."""
    global last_detections
    np_outputs = imx500.get_outputs(request.get_metadata())
    if np_outputs is None:
        return last_detections
    np_output = np.squeeze(np_outputs)
    last_detections = [
        Classification(index, np_output[index]) for index in np.argsort(-np_output)
    ]  # sort all indices by their scores
    return last_detections


def draw_classification_results(
    request: CompletedRequest, results: List[Classification], stream: str = "main"
):
    """Draw the classification results for this request onto the ISP output."""
    with MappedArray(request, stream) as m:
        text_left, text_top = 0, 0
        # drawing labels (in the ROI box if exist)
        for index, result in enumerate(results):
            label = get_label(request, idx=result.idx)
            text = f"{label}: {result.score:.3f}"
            cv2.putText(
                m.array,
                text,
                (text_left + 5, text_top + 15 + index * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path of the model")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument(
        "--labels",
        type=str,
        default="assets/labels.txt",
        help="Path to the labels file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(controls={"FrameRate": args.fps}, buffer_count=28)
    picam2.start(config, show_preview=True)
    # Register the callback to parse and draw classification results
    picam2.pre_callback = parse_and_draw_classification_results

    while True:
        time.sleep(0.5)
