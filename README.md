# aitrios-rpi-model-zoo

## Notice

### Security

Please read the Site Policy of GitHub and understand the usage conditions.

## Overview

This repository provides sample AI models and inference scripts for Raspberry Pi AI Camera Module.

## Usage

Python demo is in `examples` folder.

The `imx500_classification_demo.py` application is an example for the classification task and includes the following arguments:

- "--model", type=str, help="Path of the model"

The `imx500_object_detection_demo.py` application is an example for the object detection task and includes the following arguments:

- "--model", type=str, help="Path of the model"
- "--bbox-normalization", action="store_true", help="Normalize bbox"
- "--swap-tensors", action="store_true", help="Swap tensor 1 and 2"
- "--threshold", type=float, default=0.55, help="Detection threshold"

The `imx500_segmentation_demo.py` application is an example for the segmentation task and uses the deeplabv3plus model.

### Examples

- `python imx500_classification_demo.py --model networks/imx500_network_mobilenet_v2.rpk`
- `python imx500_object_detection_demo.py --model networks/imx500_network_efficientdet_lite0_pp.rpk --bbox-normalization`
- `python imx500_segmentation_demo.py`

## Available models

In the table below you can find information regarding all models in the model zoo.  
Please use the following models or datasets in accordance with the licenses of their respective sources.

| Task                  | Model                 | Dataset           | Float model accuracy | Quantized model accuracy | Input resolution | Original model repository                                                       | Task                        | URL for Images/Annotation                                                                                                                                                                                             |
| :-------------------- | :-------------------- | :---------------- | :------------------- | :----------------------- | :--------------- | :------------------------------------------------------------------------------ | :-------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Classification        | mobilenet_v2          | Imagenet_1k       | 0.74                 | 0.726                    | 224x224          | [Keras Applications](https://keras.io/api/applications/)                        | Classification (1000 class) | https://www.image-net.org/                                                                                                                                                                                            |
| Object detection      | efficientdet_lite0_pp | COCO (mAP)        | 0.25                 | 0.252                    | 320x320          | [efficientdet-pytorch](https://github.com/rwightman/efficientdet-pytorch)       | Object Detection (80 class) | http://images.cocodataset.org/zips/train2017.zip (19GB, 118k images) <br> http://images.cocodataset.org/zips/val2017.zip (1GB, 5k images) <br> http://images.cocodataset.org/annotations/annotations_trainval2017.zip |
| Semantic Segmentation | Deeplabv3plus         | PASCAL VOC (mIOU) | 0.72                 | 0.721                    | 320x320          | [Tensorflow](https://github.com/tensorflow/models/tree/master/research/deeplab) | Semantic Segmentation       | http://host.robots.ox.ac.uk/pascal/VOC/                                                                                                                                                                               |