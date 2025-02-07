import time
import cv2
import numpy as np

from picamera2.devices.imx500 import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)
from picamera2 import MappedArray, Picamera2


THRESHOLD = 0.55
IOU = 0.65
MAX_DETECTION = 10

last_detections = []
with open("../../../assets/coco_labels_80.txt", "r") as f:
    labels = f.read().split("\n")


class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def parse_and_draw_detections(request):
    """Analyse the detected objects in the output tensor and draw them on the main output image."""
    detections = parse_detections(request.get_metadata())
    draw_detections(request, detections)


def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP out."""
    global last_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    boxes, scores, classes = \
        postprocess_nanodet_detection(outputs=np_outputs[0], conf=THRESHOLD, iou_thres=IOU, max_out_dets=MAX_DETECTION)[0]
    from picamera2.devices.imx500.postprocess import scale_boxes
    boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)

    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score > THRESHOLD
    ]
    return last_detections


def draw_detections(request, detections, stream="main"):
    """Draw the detections for this request onto the ISP output."""
    with MappedArray(request, stream) as m:
        for detection in detections:
            x, y, w, h = detection.box
            label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"
            cv2.putText(m.array, label, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 0, 255, 0))
        b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
        cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))

if __name__ == "__main__":
    # This must be called before instantiation of Picamera2
    imx500 = IMX500("network.rpk")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(controls={"FrameRate": 30}, buffer_count=28)
    
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)
    imx500.set_auto_aspect_ratio()
    picam2.pre_callback = parse_and_draw_detections
    while True:
        time.sleep(0.5)
