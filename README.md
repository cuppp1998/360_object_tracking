# 360_object_tracking

This is a package used for object detection, object tracking and overtaking behaviour detection on panoramic (360) videos of equirectangular projection, which is implemented according to Jingwei Guo's thesis.

[YOLO v5](https://github.com/ultralytics/yolov5) and [Faster RCNN](https://github.com/facebookresearch/detectron2) models pre-trained on COCO dataset are used as the detectors in this package. Projection transformation from equirectangular to perspective is realized using [Perspective-and-Equirectangular](https://github.com/timy90022/Perspective-and-Equirectangular)
 and the implementation of DeepSORT was adapted from [HERE](https://github.com/ZQPei/deep_sort_pytorch).

## Dependencies and Installation
The library should be run under Python 3.3+ with the following libraries installed:

[detectron2 (version updated before Aug 5, 2022 only)](https://github.com/facebookresearch/detectron2/tree/5aeb252b194b93dc2879b4ac34bc51a31b5aee13)

[torch](https://github.com/pytorch/pytorch)

[torchvision](https://github.com/pytorch/pytorch)

[torchaudio](https://github.com/pytorch/pytorch)

[numpy](https://github.com/numpy/numpy)

[scipy](https://github.com/scipy/scipy)

[opencv-python](https://github.com/opencv/opencv-python)

[pillow](https://github.com/python-pillow/Pillow)

1. To install the dependencies, first run the following command in a new conda environment:
```
pip install -r requirements.txt
```
2. Clone the repository:
```
git clone https://github.com/cuppp1998/360_object_tracking.git
```
