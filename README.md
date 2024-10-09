# aitrios-rpi-model-zoo

## Notice

### Security

Please read the Site Policy of GitHub and understand the usage conditions.

## Overview

This repository provides sample AI models and inference scripts for Raspberry Pi AI Camera Module.

## Usage

Python demo is in `models` folder.

### Brain Builder

The model (\*.rpk), label, and config files are not included in this repository. These are output files from Brain Builder.

#### Anomaly Recognizer

In `anomaly-detection/brainbuilder`

```
python app.py --model YOUR_MODEL.rpk --config neurala_hifi_ppl_parameters.json --fps 15
```

#### Classifier

In `classification/brainbuilder`

```
python app.py --model YOUR_MODEL.rpk --labels labels.txt --fps 15
```

#### Detector

In `object-detection/brainbuilder`

```
python app.py --model YOUR_MODEL.rpk --labels labels.txt --fps 15
```

### The other Python demos

#### Classification

In `classification/mobilenet_v2`

```
python app.py
```

#### Object Detection

In `object-detection/efficientdet_lite0_pp`

```
python app.py
```

#### Segmentation

In `segmentation/deeplabv3plus`

```
python app.py
```

## Available models

In the table below you can find information regarding all models in the model zoo.  
Please use the following models or datasets in accordance with the licenses of their respective sources.

| Task                  | Model                 | Dataset           | Float model accuracy | Quantized model accuracy | Input resolution | Original model repository                                                       | URL for Images/Annotation                                                                                                                                                                                             |
| :-------------------- | :-------------------- | :---------------- | :------------------- | :----------------------- | :--------------- | :------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Classification        | mobilenet_v2          | Imagenet_1k       | 0.74                 | 0.726                    | 224x224          | [Keras Applications](https://keras.io/api/applications/)                        | https://www.image-net.org/                                                                                                                                                                                            |
| Object detection      | efficientdet_lite0_pp | COCO (mAP)        | 0.25                 | 0.252                    | 320x320          | [efficientdet-pytorch](https://github.com/rwightman/efficientdet-pytorch)       | http://images.cocodataset.org/zips/train2017.zip (19GB, 118k images) <br> http://images.cocodataset.org/zips/val2017.zip (1GB, 5k images) <br> http://images.cocodataset.org/annotations/annotations_trainval2017.zip |
| Semantic Segmentation | Deeplabv3plus         | PASCAL VOC (mIOU) | 0.72                 | 0.721                    | 320x320          | [Tensorflow](https://github.com/tensorflow/models/tree/master/research/deeplab) | http://host.robots.ox.ac.uk/pascal/VOC/                                                                                                                                                                               |
