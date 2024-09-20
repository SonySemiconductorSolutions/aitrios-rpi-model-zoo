import argparse
import time
from functools import lru_cache

import cv2
from picamera2 import MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500

last_detections = []


class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        obj_scaled = imx500.convert_inference_coords(coords, metadata, picam2)
        self.box = (obj_scaled.x, obj_scaled.y, obj_scaled.width, obj_scaled.height)


def parse_and_draw_detections(request):
    """Analyse the detected objects in the output tensor and draw them on the main output image."""
    detections = parse_detections(request.get_metadata())
    draw_detections(request, detections)


def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP out."""
    global last_detections
    threshold = args.threshold

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_w_h()
    if np_outputs is None:
        return last_detections
    boxes, scores, classes = np_outputs[0][0], np_outputs[2][0], np_outputs[1][0]
    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score >= threshold
    ]
    return last_detections


@lru_cache
def get_labels():
    with open(args.labels, "r") as f:
        labels = f.read().split("\n")

    return labels


def draw_detections(request, detections, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    labels = get_labels()
    with MappedArray(request, stream) as m:
        for detection in detections:
            x, y, w, h = detection.box
            label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"
            cv2.putText(
                m.array,
                label,
                (x + 5, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 0, 255, 0))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path of the model")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection threshold")
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
    picam2.pre_callback = parse_and_draw_detections

    while True:
        time.sleep(0.5)
