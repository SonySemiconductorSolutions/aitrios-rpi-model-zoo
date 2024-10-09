import argparse
import json
import os
import time
import tkinter

import cv2
import numpy as np
from picamera2 import CompletedRequest, Picamera2
from picamera2.devices.imx500 import IMX500

WINDOW_SIZE = (640, 480)
image_thresh = None
pixel_thresh = None


def get_thresh():
    """Get Threshold for anomaly visualization."""
    global image_thresh, pixel_thresh
    if (image_thresh is None) or (pixel_thresh is None):
        if os.path.exists(args.config):
            with open(args.config, "r") as f:
                data = json.load(f)
            image_thresh = data["imageThreshold"]
            pixel_thresh = data["pixelThreshold"]
        else:
            image_thresh = 0
            pixel_thresh = 0


def on_pixel_thresh_changed(event):
    """pixel_thresh is changed on Setting Window."""
    global pixel_thresh, pixel_thresh_slider
    pixel_thresh = pixel_thresh_slider.get()


def on_image_thresh_changed(event):
    """image_thresh is changed on Setting Window."""
    global image_thresh, image_thresh_slider
    image_thresh = image_thresh_slider.get()


def create_and_draw_mask(request: CompletedRequest):
    """Create masks from the output tensor and draw them on the main output image."""
    global result_value, anomaly_value
    score, mask = create_mask(request)
    draw_mask(request, mask)
    result = "Anomaly" if (score > image_thresh) else "Normal"
    result_value["text"] = result
    anomaly_value["text"] = f"{score:.3f}"


def create_mask(request: CompletedRequest) -> tuple[float, np.ndarray]:
    """Create masks from the output tensor, scaled to the ISP out."""
    global pixel_thresh
    np_outputs = imx500.get_outputs(metadata=request.get_metadata())
    np_output = np.squeeze(np_outputs)
    image_score = np_output[0, 0, 1]
    pixel_score = np_output[:, :, 0]

    mask = np.zeros(pixel_score.shape + (4,), dtype=np.uint8)
    mask[pixel_score >= pixel_thresh, 0] = 255
    mask[pixel_score >= pixel_thresh, 3] = 150
    mask = cv2.resize(mask, WINDOW_SIZE, interpolation=cv2.INTER_NEAREST)
    return image_score, mask


def draw_mask(request: CompletedRequest, mask: np.ndarray):
    """Draw the masks for this request onto the ISP output."""
    if mask is not None:
        picam2.set_overlay(mask)


def on_click(event):
    """If clicked focus on that UI object (Slider can controlled by cursor key when focus is on)"""
    event.widget.focus_set()


def create_setting_window():
    """Anomaly Setting Window"""
    global pixel_thresh, pixel_thresh_slider
    global image_thresh, image_thresh_slider
    global result_value, anomaly_value

    setting_window = tkinter.Tk()
    setting_window.title("Settings")

    result_label = tkinter.Label(text="Result:")
    anomaly_label = tkinter.Label(text="Anomaly score:")
    result_value = tkinter.Label()
    anomaly_value = tkinter.Label()
    image_thresh_label = tkinter.Label(text="imageThreshold:")
    pixel_thresh_label = tkinter.Label(text="pixelThreshold:")

    slider_len = 250
    slider_width = 15

    pixel_thresh_slider = tkinter.Scale(
        setting_window,
        orient=tkinter.HORIZONTAL,
        from_=0.0,
        to=1.0,
        resolution=0.001,
        length=slider_len,
        width=slider_width,
        command=on_pixel_thresh_changed,
        takefocus=True,
    )
    pixel_thresh_slider.set(pixel_thresh)
    pixel_thresh_slider.bind("<Button-1>", on_click)  # click on focus

    image_thresh_slider = tkinter.Scale(
        setting_window,
        orient=tkinter.HORIZONTAL,
        from_=0.0,
        to=1.0,
        resolution=0.001,
        length=slider_len,
        width=slider_width,
        command=on_image_thresh_changed,
        takefocus=True,
    )
    image_thresh_slider.set(image_thresh)
    image_thresh_slider.bind("<Button-1>", on_click)  # click on focus

    # layout
    setting_window.geometry("400x150")
    result_label.grid(row=0, column=0, sticky=tkinter.E)
    result_value.grid(row=0, column=1, sticky=tkinter.W)
    anomaly_label.grid(row=1, column=0, sticky=tkinter.E)
    anomaly_value.grid(row=1, column=1, sticky=tkinter.W)
    pixel_thresh_label.grid(row=2, column=0, sticky=tkinter.E)
    pixel_thresh_slider.grid(row=2, column=1, sticky=tkinter.W)
    image_thresh_label.grid(row=3, column=0, sticky=tkinter.E)
    image_thresh_slider.grid(row=3, column=1, sticky=tkinter.W)

    return setting_window


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path of the model")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    parser.add_argument(
        "--config",
        type=str,
        default="assets/neurala_hifi_ppl_parameters.json",
        help="Path to the threshold config file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    get_thresh()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(controls={"FrameRate": args.fps}, buffer_count=28)

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)
    setting_window = create_setting_window()

    picam2.pre_callback = create_and_draw_mask
    setting_window.mainloop()

    while True:
        time.sleep(0.5)
