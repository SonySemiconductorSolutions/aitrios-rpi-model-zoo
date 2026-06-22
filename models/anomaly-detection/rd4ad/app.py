import argparse
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter
from typing import List
from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices import IMX500


previous_anomaly_map = None


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default="network.rpk")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay the anomaly map directly on the input image"
    )
    return parser.parse_args()


def cal_anomaly_map(
        fs_list: List[np.ndarray],
        ft_list: List[np.ndarray],
        out_size=256, amap_mode='mul') -> np.ndarray:
    anomaly_map = (
        np.ones([out_size, out_size])
        if amap_mode == 'mul'
        else np.zeros([out_size, out_size])
    )
    for fs, ft in zip(fs_list, ft_list):
        fs, ft = torch.tensor(fs).squeeze(0), torch.tensor(ft).squeeze(0)
        a_map = 1 - F.cosine_similarity(fs, ft,
                                        dim=0).unsqueeze(0).unsqueeze(0)
        a_map = F.interpolate(
            a_map, size=out_size, mode='bilinear', align_corners=True
        )[0, 0, :, :].cpu().numpy()
        anomaly_map = (
            anomaly_map * a_map
            if amap_mode == 'mul'
            else anomaly_map + a_map
        )
    return anomaly_map


def min_max_norm(image: np.ndarray) -> np.ndarray:
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def create_anomaly_map(request: CompletedRequest) -> np.ndarray:
    """Generate the anomaly map from the output tensors."""
    global previous_anomaly_map
    output_tensors = imx500.get_outputs(request.get_metadata())
    if output_tensors is None:
        if previous_anomaly_map is None:
            previous_anomaly_map = np.zeros(
                imx500.get_input_size(), dtype=np.uint8)
        return previous_anomaly_map

    fs_list = output_tensors[:3]  # Encoder output
    ft_list = output_tensors[3:]  # Decoder output
    out_size = imx500.get_input_size()[0]
    anomaly_map = gaussian_filter(cal_anomaly_map(
        fs_list, ft_list, out_size=out_size), sigma=4)
    anomaly_map = (min_max_norm(anomaly_map) * 255).astype(np.uint8)
    previous_anomaly_map = anomaly_map
    return anomaly_map


def draw_anomaly_map(
        request: CompletedRequest,
        anomaly_map: np.ndarray,
        overlay_mode: bool):
    """Draw the anomaly map on the main output image."""
    heatmap = cv2.applyColorMap(anomaly_map.astype(np.uint8), cv2.COLORMAP_JET)

    input_width, input_height = imx500.get_input_size()

    if overlay_mode:
        with MappedArray(request, "main") as m:
            heatmap_resized = cv2.resize(
                heatmap, (m.array.shape[1], m.array.shape[0]))
            overlay = cv2.addWeighted(cv2.cvtColor(
                m.array, cv2.COLOR_RGBA2BGR), 0.6, heatmap_resized, 0.4, 0)
            m.array[:] = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGBA)
    else:
        overlay = np.zeros((input_height, input_width, 4), dtype=np.uint8)
        resized_heatmap = cv2.resize(
            heatmap, (input_width // 4, input_height // 4))
        x_offset, y_offset = input_width - \
            resized_heatmap.shape[1] - 10, input_height - \
            resized_heatmap.shape[0] - 10
        overlay[
            y_offset:y_offset + resized_heatmap.shape[0],
            x_offset:x_offset + resized_heatmap.shape[1]
        ] = cv2.cvtColor(
            resized_heatmap,
            cv2.COLOR_BGR2RGBA
        )
        picam2.set_overlay(overlay)


def create_and_draw_anomaly_map(request: CompletedRequest):
    """Generate and draw the anomaly map in a single function for callback."""
    anomaly_map = create_anomaly_map(request)
    draw_anomaly_map(request, anomaly_map, args.overlay)


if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        controls={"FrameRate": args.fps or 30}, buffer_count=12)
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)
    picam2.pre_callback = create_and_draw_anomaly_map

    while True:
        time.sleep(0.5)
