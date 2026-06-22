# aitrios-rpi-model-zoo

## Notice

### Security

Please read the Site Policy of GitHub and understand the usage conditions.

## Overview

This repository provides sample AI models and inference scripts for Raspberry Pi AI Camera Module.

## Usage

Before running the Python demos, please make sure to install OpenCV:
```bash
sudo apt install python3-opencv python3-munkres
```

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

### Training Samples
The model (\*.rpk), label are not included in this repository. These are output files from [aitrios-rpi-training-samples](https://github.com/SonySemiconductorSolutions/aitrios-rpi-training-samples)

In `anomaly-detection/rd4ad`

```
python app.py --model YOUR_MODEL.rpk --fps 15 --overlay
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

#### Pose Estimation

In `pose_estimation/higherhrnet_coco`

```
python app.py
```

## Available models

In the table below you can find information regarding all models in the model zoo.  
Please use the following models or datasets in accordance with the licenses of their respective sources.

| Task                  | Model                 | Dataset           | Float model accuracy | Quantized model accuracy | Input resolution | Original model repository | License                                                      | URL for Images/Annotation                                                                                                                                                                                             |
| :-------------------- | :-------------------- | :---------------- | :------------------- | :----------------------- | :--------------- | :--------- |:------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Classification        | efficientnet_lite0          | Imagenet_1k       | 0.753                 | 0.753                    | 224x224          | [trimm repository](https://github.com/huggingface/pytorch-image-models)                        | [models/classification/ efficientnet_lite0/LICENSE.md](models/classification/efficientnet_lite0/LICENSE.md)          |  https://www.image-net.org/ 
| Classification        | efficientnetv2_b0          | Imagenet_1k       | 0.764                 | 0.767                    | 224x224          | [Keras Applications](https://keras.io/api/applications/)                        | [models/classification/ efficientnetv2_b0/LICENSE.md](models/classification/efficientnetv2_b0/LICENSE.md)            | https://www.image-net.org/ 
| Classification        | efficientnetv2_b1          | Imagenet_1k       | 0.769                 | 0.777                    | 240x240          | [Keras Applications](https://keras.io/api/applications/)                        | [models/classification/ efficientnetv2_b1/LICENSE.md](models/classification/efficientnetv2_b1/LICENSE.md)          | https://www.image-net.org/ 
| Classification        | efficientnetv2_b2          | Imagenet_1k       | 0.779                 | 0.770                    | 260x260          | [Keras Applications](https://keras.io/api/applications/)                        | [models/classification/ efficientnetv2_b2/LICENSE.md](models/classification/efficientnetv2_b2/LICENSE.md)          | https://www.image-net.org/ 
| Classification        | efficientnet_bo          | Imagenet_1k       | 0.739                 | 0.721                    | 224x224          | [Torchvision](https://github.com/pytorch/vision)                        | [models/classification/ efficientnet_bo/LICENSE.md](models/classification/efficientnet_bo/LICENSE.md)           | https://www.image-net.org/ 
| Classification        | resnet18          | Imagenet_1k       | 0.685                 | 0.686                    | 224x224          | [trimm repository](https://github.com/huggingface/pytorch-image-models)                        | [models/classification/ resnet18/LICENSE.md](models/classification/resnet18/LICENSE.md)           | https://www.image-net.org/
| Classification        | regnety_002          | Imagenet_1k       | 0.696                 | 0.694                    | 224x224          | [trimm repository](https://github.com/huggingface/pytorch-image-models)                        | [models/classification/ regnety_002/LICENSE.md](models/classification/regnety_002/LICENSE.md)           | https://www.image-net.org/ 
| Classification        | regnetx_002          | Imagenet_1k       | 0.682                 | 0.684                    | 224x224          | [trimm repository](https://github.com/huggingface/pytorch-image-models)                        | [models/classification/ regnetx_002/LICENSE.md](models/classification/regnetx_002/LICENSE.md)          | https://www.image-net.org/ 
| Classification        | regnety_004          | Imagenet_1k       | 0.734                 | 0.738                    | 224x224          | [trimm repository](https://github.com/huggingface/pytorch-image-models)                        | [models/classification/ regnety_004/LICENSE.md](models/classification/regnety_004/LICENSE.md)          | https://www.image-net.org/ 
| Classification        | mobilevit_xs          | Imagenet_1k       | 0.724                 | 0.723                    | 256x256          | [trimm repository](https://github.com/huggingface/pytorch-image-models)                        | [models/classification/ mobilevit_xs/LICENSE.md](models/classification/mobilevit_xs/LICENSE.md)         | https://www.image-net.org/
| Classification        | mobilevit_xxs          | Imagenet_1k       | 0.674                 | 0.674                    | 256x256          | [trimm repository](https://github.com/huggingface/pytorch-image-models)                        | [models/classification/ mobilevit_xxs/LICENSE.md](models/classification/mobilevit_xxs/LICENSE.md)        | https://www.image-net.org/  
| Classification        | levit_128s          | Imagenet_1k       | 0.583                 | 0.623                    | 224x224          | [Facebook research](https://github.com/facebookresearch/LeViT)                        | [models/classification/ levit_128s/LICENSE.md](models/classification/levit_128s/LICENSE.md)          | https://www.image-net.org/  
| Classification        | mobilenet_v2          | Imagenet_1k       | 0.74                 | 0.726                    | 224x224          | [Keras Applications](https://keras.io/api/applications/)                        | [models/classification/ mobilenet_v2/LICENSE.md](models/classification/mobilenet_v2/LICENSE.md)          | https://www.image-net.org/
| Classification        | shufflenet_v2_x1_5          | Imagenet_1k       | 0.725                 | 0.722                    | 224x224          | [trimm repository](https://github.com/huggingface/pytorch-image-models)                        | [models/classification/ shufflenet_v2_x1_5/LICENSE.md](models/classification/shufflenet_v2_x1_5/LICENSE.md)          | https://www.image-net.org/ 
| Classification        | mnasnet1.0         | Imagenet_1k       | 0.731                 | 0.732                    | 224x224          | [trimm repository](https://github.com/huggingface/pytorch-image-models)                        | [models/classification/ mnasnet1.0/LICENSE.md](models/classification/mnasnet1.0/LICENSE.md)         | https://www.image-net.org/  
| Classification        | squeezenet1.0         | Imagenet_1k       | 0.576                 | 0.576                    | 224x224          | [trimm repository](https://github.com/huggingface/pytorch-image-models)                        | [models/classification/ squeezenet1.0/LICENSE.md](models/classification/squeezenet1.0/LICENSE.md)        | https://www.image-net.org/  
| Object Detection      | efficientdet_lite0 | COCO (mAP)        | 0.273                 | 0.274                    | 320x320          | [efficientdet-pytorch](https://github.com/rwightman/efficientdet-pytorch)       | [models/object-detection/ efficientdet_lite0/LICENSE.md](models/object-detection/efficientdet_lite0/LICENSE.md) | http://images.cocodataset.org/zips/train2017.zip (19GB, 118k images) <br> http://images.cocodataset.org/zips/val2017.zip (1GB, 5k images) <br> http://images.cocodataset.org/annotations/annotations_trainval2017.zip |
| Object Detection      | nanodet_plus_416x416_pp | COCO (mAP)        | 0.323                 | 0.32                    | 416x416          | [Nanodet](https://github.com/RangiLyu/nanodet)       | [models/object-detection/ nanodet_plus_416x416_pp/LICENSE.md](models/object-detection/nanodet_plus_416x416_pp/LICENSE.md) | http://images.cocodataset.org/zips/train2017.zip (19GB, 118k images) <br> http://images.cocodataset.org/zips/val2017.zip (1GB, 5k images) <br> http://images.cocodataset.org/annotations/annotations_trainval2017.zip |
| Object Detection      | nanodet_plus_416x416 | COCO (mAP)        | 0.332                 | 0.332                    | 416x416          | [Nanodet](https://github.com/RangiLyu/nanodet)       | [models/object-detection/ nanodet_plus_416x416/LICENSE.md](models/object-detection/nanodet_plus_416x416/LICENSE.md) | http://images.cocodataset.org/zips/train2017.zip (19GB, 118k images) <br> http://images.cocodataset.org/zips/val2017.zip (1GB, 5k images) <br> http://images.cocodataset.org/annotations/annotations_trainval2017.zip |
| Object Detection      | efficientdet_lite0_pp | COCO (mAP)        | 0.25                 | 0.252                    | 320x320          | [efficientdet-pytorch](https://github.com/rwightman/efficientdet-pytorch)       | [models/object-detection/ efficientdet_lite0_pp/LICENSE.md](models/object-detection/efficientdet_lite0_pp/LICENSE.md) | http://images.cocodataset.org/zips/train2017.zip (19GB, 118k images) <br> http://images.cocodataset.org/zips/val2017.zip (1GB, 5k images) <br> http://images.cocodataset.org/annotations/annotations_trainval2017.zip |
| Pose Estimation      | higherhrnet (coco) - multi person | COCO        | 0.187                 | 0.188                    | 288x384          | [HRnet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation) | [models/pose-estimation/ higherhrnet_coco/LICENSE.md](models/pose-estimation/higherhrnet_coco/LICENSE.md)       | http://images.cocodataset.org/zips/train2017.zip (19GB, 118k images) <br> http://images.cocodataset.org/zips/val2017.zip (1GB, 5k images) <br> http://images.cocodataset.org/annotations/annotations_trainval2017.zip |
| Semantic Segmentation | Deeplabv3plus         | PASCAL VOC (mIOU) | 0.72                 | 0.721                    | 320x320          | [Tensorflow](https://github.com/tensorflow/models/tree/master/research/deeplab) | [models/segmentation/ Deeplabv3plus/LICENSE.md](models/segmentation/Deeplabv3plus/LICENSE.md)         | http://host.robots.ox.ac.uk/pascal/VOC/                                                                                                                                                                               |
