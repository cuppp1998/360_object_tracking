# 360_object_tracking

This is a package used for object detection, object tracking and overtaking behaviour detection on panoramic (360) videos of equirectangular projection, which is implemented according to Jingwei Guo's thesis.

[YOLO v5](https://github.com/ultralytics/yolov5) and [Faster RCNN](https://github.com/facebookresearch/detectron2) models pre-trained on COCO dataset are used as the detectors in this package. Projection transformation from equirectangular to perspective is realized using [Perspective-and-Equirectangular](https://github.com/timy90022/Perspective-and-Equirectangular)
 and the implementation of DeepSORT was adapted from [HERE](https://github.com/ZQPei/deep_sort_pytorch).

## Dependencies and Installation
The library should be run under Python 3.3+ with the following libraries installed:

[detectron2 (version updated before Aug 5, 2022 only)](https://github.com/facebookresearch/detectron2/tree/5aeb252b194b93dc2879b4ac34bc51a31b5aee13)

[torch](https://github.com/pytorch/pytorch)

[torchvision](https://github.com/pytorch/pytorch)

[numpy](https://github.com/numpy/numpy)

[matplotlib](https://github.com/matplotlib/matplotlib)

[scipy](https://github.com/scipy/scipy)

[opencv-python](https://github.com/opencv/opencv-python)

[pillow](https://github.com/python-pillow/Pillow)

[pandas](https://github.com/pandas-dev/pandas)

[seaborn](https://github.com/mwaskom/seaborn)

1. First, clone the repository:
```
git clone https://github.com/cuppp1998/360_object_tracking.git
```
2. To install all the dependencies (except Detectron2), run the following command in a new conda environment:
```
cd 360_object_tracking
pip install -r requirements.txt
```
3. Since in the new versions of Detectron2 (updated after Aug 5, 2022), some APIs have been modified, here we install an old version of it:
```
pip install -e git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13#egg=detectron2
```
4. Download the pre-trained ReID network used in DeepSORT:
```
cd deep_sort/deep/checkpoint
pip install gdown
gdown 'https://drive.google.com/uc?export=download&id=1_qwTWdzT9dWNudpusgKavj_4elGgbkUN'
cd ../../../
```

## Instruction of the Main Functionalities
The implementation process of each functionality is explained in detail in [Code Explanation.ipynb](./Code%20Explanation.ipynb).
### 360 Object Detection

To realize object detection on panoramic videos of equirectangular projection, execute Object_Detection.py in the Terminal as below:
```
python Object_Detection.py [--input_video_path INPUT_VIDEO_PATH] [--output_video_path OUTPUT_VIDEO_PATH] [--classes_to_detect CLASSES_TO_DETECT] [--FOV FOV] [--THETAs THETAS] [--PHIs PHIS] [--sub_image_width SUB_IMAGE_WIDTH] [--model_type MODEL_TYPE] [--score_threshold SCORE_THRESHOLD] [--nms_threshold NMS_THRESHOLD] [--use_mymodel USE_MYMODEL]
```
The following arguments are provided:

|  Argument   | Description  | Required? | Defaults |
|  :----:  | :----:  | :----:  | :----:  |
| INPUT_VIDEO_PATH  | Path of the input video | ✔️ |  |
| OUTPUT_VIDEO_PATH  | Path of the output video | ✔️ |  |
| CLASSES_TO_DETECT  | Index numbers of the categories to detect in the COCO dataset |  | [0, 1, 2, 3, 5, 7, 9] |
| FOV  | Field of view of the sub images |  | 120 |
| THETAS  | A list which contains the theta of each sub image (The length should be the same as the number of sub images) |  | [0, 90, 180, 270] |
| PHIS  | A list which contains the Phi of each sub image (The length should be the same as the number of sub images) |  | [-10, -10, -10, -10] |
| SUB_IMAGE_WIDTH  | Width (or height) of the sub images |  | 640 |
| MODEL_TYPE  | A string that determines which detector to use ("YOLO" or "Faster RCNN") |  |"YOLO" |
| SCORE_THRESHOLD  | The threshold of the confidence score |  | 0.4 |
| NMS_THRESHOLD  | The threshold of the Non Maximum Suppression |  | 0.45 |
| USE_MYMODEL  | A boolean value which determines whether to use the improved object detection model, if False, instead of being split into 4 parts, the image will be detected as a whole |  | True |

### 360 Object Tracking

To realize object tracking on panoramic videos of equirectangular projection, execute Object_Tracking.py in the Terminal as below:
```
python Object_Tracking.py [--input_video_path INPUT_VIDEO_PATH] [--output_video_path OUTPUT_VIDEO_PATH] [--MOT_text_path MOT_TEXT_PATH] [--prevent_different_classes_match PREVENT_DIFFERENT_CLASSES_MATCH] [--match_across_boundary MATCH_ACROSS_BOUNDARY] [--classes_to_detect CLASSES_TO_DETECT] [--FOV FOV] [--THETAs THETAS] [--PHIs PHIS] [--sub_image_width SUB_IMAGE_WIDTH] [--model_type MODEL_TYPE] [--score_threshold SCORE_THRESHOLD] [--nms_threshold NMS_THRESHOLD] [--use_mymodel USE_MYMODEL]
```
The following arguments are provided:

|  Argument   | Description  | Required? | Defaults |
|  :----:  | :----:  | :----:  | :----:  |
| INPUT_VIDEO_PATH  | Path of the input video | ✔️ |  |
| OUTPUT_VIDEO_PATH  | Path of the output video | ✔️ |  |
| MOT_TEXT_PATH  | Path of the output txt file which stores all the MOT tracking results | ✔️ |  |
| PREVENT_DIFFERENT_CLASSES_MATCH  | A boolean value which determines whether to use the support for multiple categories in DeepSORT |  | True |
| MATCH_ACROSS_BOUNDARY  | A boolean value which determines whether to use the support for boundary continuity in DeepSORT |  | True |
| CLASSES_TO_DETECT  | Index numbers of the categories to detect in the COCO dataset |  | [0, 1, 2, 3, 5, 7, 9] |
| FOV  | Field of view of the sub images |  | 120 |
| THETAS  | A list which contains the theta of each sub image (The length should be the same as the number of sub images) |  | [0, 90, 180, 270] |
| PHIS  | A list which contains the Phi of each sub image (The length should be the same as the number of sub images) |  | [-10, -10, -10, -10] |
| SUB_IMAGE_WIDTH  | Width (or height) of the sub images |  | 640 |
| MODEL_TYPE  | A string that determines which detector to use ("YOLO" or "Faster RCNN") |  |"YOLO" |
| SCORE_THRESHOLD  | The threshold of the confidence score |  | 0.4 |
| NMS_THRESHOLD  | The threshold of the Non Maximum Suppression |  | 0.45 |
| USE_MYMODEL  | A boolean value which determines whether to use the improved object detection model, if False, instead of being split into 4 parts, the image will be detected as a whole |  | True |

### 360 Overtaking Behaviour Detection

To realize overtaking behaviour detection on panoramic videos of equirectangular projection, execute Overtaking_Detection.py in the Terminal as below:
```
python Overtaking_Detection.py [--input_video_path INPUT_VIDEO_PATH] [--output_video_path OUTPUT_VIDEO_PATH] [--mode MODE] [--prevent_different_classes_match PREVENT_DIFFERENT_CLASSES_MATCH] [--match_across_boundary MATCH_ACROSS_BOUNDARY] [--classes_to_detect CLASSES_TO_DETECT] [--classes_to_detect_movement CLASSES_TO_DETECT_MOVEMENT] [--size_thresholds SIZE_THRESHOLDS] [--FOV FOV] [--THETAs THETAS] [--PHIs PHIS] [--sub_image_width SUB_IMAGE_WIDTH] [--model_type MODEL_TYPE] [--score_threshold SCORE_THRESHOLD] [--nms_threshold NMS_THRESHOLD] [--use_mymodel USE_MYMODEL]
```
The following arguments are provided:

|  Argument   | Description  | Required? | Defaults |
|  :----:  | :----:  | :----:  | :----:  |
| INPUT_VIDEO_PATH  | Path of the input video | ✔️ |  |
| OUTPUT_VIDEO_PATH  | Path of the output video | ✔️ |  |
| MODE | A string that determines which kind of overtaking behaviour to detect, "Confirmed" or "Unconfirmed" |  | "Confirmed" |
| PREVENT_DIFFERENT_CLASSES_MATCH  | A boolean value which determines whether to use the support for multiple categories in DeepSORT |  | True |
| MATCH_ACROSS_BOUNDARY  | A boolean value which determines whether to use the support for boundary continuity in DeepSORT |  | True |
| CLASSES_TO_DETECT  | Index numbers of the categories to detect in the COCO dataset |  | [0, 1, 2, 3, 5, 7, 9] |
| CLASSES_TO_DETECT_MOVEMENT  | Index numbers of the categories for movement detection in the COCO dataset, which should be a subset of classes_to_detect |  | [2, 5, 7] |
| SIZE_THRESHOLDS  | A set of size thresholds which should share the same length with classes_to_detect_movement, if the size of a track of a certain class is larger than the corresponding threshold, then it is considered as close to the user |  | [500 * 500, 900 * 900, 600 * 600] |
| FOV  | Field of view of the sub images |  | 120 |
| THETAS  | A list which contains the theta of each sub image (The length should be the same as the number of sub images) |  | [0, 90, 180, 270] |
| PHIS  | A list which contains the Phi of each sub image (The length should be the same as the number of sub images) |  | [-10, -10, -10, -10] |
| SUB_IMAGE_WIDTH  | Width (or height) of the sub images |  | 640 |
| MODEL_TYPE  | A string that determines which detector to use ("YOLO" or "Faster RCNN") |  |"YOLO" |
| SCORE_THRESHOLD  | The threshold of the confidence score |  | 0.4 |
| NMS_THRESHOLD  | The threshold of the Non Maximum Suppression |  | 0.45 |
| USE_MYMODEL  | A boolean value which determines whether to use the improved object detection model, if False, instead of being split into 4 parts, the image will be detected as a whole |  | True |

## Examples
For better understanding, several examples of using this package is listed as below:

1. To <b>detect</b> the bicycles, cars and motorbikes ([1, 2, 3] in COCO) in a video called test.mp4 with the <b>original</b> Faster RCNN and output the result video as test_object_detection.mp4, run the following command:
```
python Object_Detection.py --input_video_path test.mp4 --output_video_path test_object_detection.mp4 --use_mymodel False --classes_to_detect 1 2 3 --model_type "Faster RCNN"
```

2. To <b>track</b> the people and cars ([0, 2] in COCO) in a video called test.mp4 with the improved YOLO v5 whose input resolution is 1280, and to output the result video and MOT texts as test_object_tracking.mp4 and test_object_tracking.txt, run the following command:
```
python Object_Tracking.py --input_video_path test.mp4 --output_video_path test_object_tracking.mp4 --MOT_text_path test_object_tracking.txt --classes_to_detect 0 2 --sub_image_width 1280
```

3. To track people, bicycles, cars, motorbikes, buses, trucks and traffic lights ([0, 1, 2, 3, 5, 7, 9] in COCO) in a video called test.mp4 and <b>detect the close unconfirmed overtakes (size<160000)</b> of only cars with the improved YOLO v5 whose input resolution is 640, and to output the result video as test_overtaking_detection.mp4, run the following command:
```
python Overtaking_Detection.py --input_video_path test.mp4 --output_video_path test_overtaking_detection.mp4 --mode 'Unconfirmed' --classes_to_detect_movement 2 --size_thresholds 160000
```
