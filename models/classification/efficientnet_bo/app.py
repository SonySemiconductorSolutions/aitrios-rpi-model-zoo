import time
from typing import List
import cv2
import numpy as np

from picamera2.devices.imx500 import IMX500
from picamera2 import MappedArray, Picamera2, CompletedRequest
from picamera2.devices.imx500.postprocess import softmax
from picamera2.devices.imx500 import NetworkIntrinsics

last_detections = []

with open("../../../assets/imagenet_labels.txt", "r") as f:
    labels = f.read().splitlines()
    labels = labels[1:]


class Classification:
    def __init__(self, idx: int, score: float):
        """Create a Classification object, recording the idx and score."""
        self.idx = idx
        self.score = score


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
    np_output = np_outputs[0]
    np_output = softmax(np_output)
    top_indices = np.argpartition(-np_output, 3)[:3]  # Get top 3 indices with the highest scores
    top_indices = top_indices[np.argsort(-np_output[top_indices])]  # Sort the top 3 indices by their scores
    last_detections = [Classification(index, np_output[index]) for index in top_indices]
    return last_detections


def draw_classification_results(request: CompletedRequest, results: List[Classification], stream: str = "main"):
    """Draw the classification results for this request onto the ISP output."""
    with MappedArray(request, stream) as m:
        # drawing roi box
        b_x, b_y, b_w, b_h  = imx500.get_roi_scaled(request)
        cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1)
        cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))
        text_left, text_top = b_x, b_y + 20
        # drawing labels (in the ROI box if exist)
        for index, result in enumerate(results):
            label = labels[result.idx]
            text = f"{label}: {result.score:.3f}"
            cv2.putText(m.array, text, (text_left + 5, text_top + 15 + index * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


if __name__ == "__main__":
    # This must be called before instantiation of Picamera2
    imx500 = IMX500("network.rpk")
    intrinsics = imx500.network_intrinsics
    print("intrinsics : ", intrinsics)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(controls={"FrameRate": 30}, buffer_count=28)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)
    imx500.set_auto_aspect_ratio()
    # Register the callback to parse and draw classification results
    picam2.pre_callback = parse_and_draw_classification_results

    while True:
        time.sleep(0.5)
