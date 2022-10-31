# import some common libraries
import numpy as np
import os, cv2
import torch
import torchvision

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# import the Perspective_and_Equirectangular library
import lib.Equirec2Perspec as E2P


import time
from detectron2.layers import batched_nms

# function used to load a YOLO or Faster RCNN model according to the users' demands
def load_model(model_type, input_size=1280, score_threshold=0.4, nms_threshold=0.45):

    # first get the default config
    cfg = get_cfg()

    # choose a model from detectron2's model zoo
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.INPUT.MAX_SIZE_TEST = input_size  # set the size of the input images
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        score_threshold  # set the threshold of the confidence score
    )
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_threshold  # set the NMS threshold

    # set the device to use (GPU or CPU)
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
    else:
        cfg.MODEL.DEVICE = "cpu"

        # only work on apple m1 mac
        # cfg.MODEL.DEVICE = 'mps'

    # create a predictor instance with the config above
    predictor_faster_RCNN = DefaultPredictor(cfg)

    # choose a model from YOLO v5 family
    predictor_YOLO = torch.hub.load("ultralytics/yolov5", "yolov5m6")
    predictor_YOLO.conf = score_threshold  # set the threshold of the confidence score
    predictor_YOLO.iou = nms_threshold  # set the NMS threshold
    predictor_YOLO.agnostic = True  # NMS class-agnostic (i.e., only the bboxes with the same category can be eliminated after NMS)

    if model_type == "Faster RCNN":
        return predictor_faster_RCNN, cfg
    else:
        return predictor_YOLO, cfg


# function used to split the equirectangular image into several sub images which are in perspective projection
def equir2pers(input_img, FOV, THETAs, PHIs, output_height, output_width):
    equ = E2P.Equirectangular(input_img)  # Load the equirectangular image

    # set where to save the outputs
    output_dir = "./output_sub/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # maps which define the projection from equirectangular to perspective
    lon_maps = []
    lat_maps = []
    imgs = []  # output images

    # for each sub image
    for i in range(len(PHIs)):
        img1, lon_map1, lat_map1 = equ.GetPerspective(
            FOV, THETAs[i], PHIs[i], output_height, output_width
        )
        # save the outputs
        output1 = output_dir + str(i) + ".png"
        cv2.imwrite(output1, img1)
        lon_maps.append(lon_map1)
        lat_maps.append(lat_map1)
        imgs.append(img1)

    return lon_maps, lat_maps, imgs


# function used to reproject the bboxes on the sub images (perspective) to the original image (equirectangular)
# and find the bboxes whose left/right border is tangent to a border of the sub image (i.e., distance < threshold_of_boundary)
def reproject_bboxes(
    bboxes,
    lon_map_original,
    lat_map_original,
    classes,
    scores,
    interval,
    num_of_subimage,
    input_video_width,
    input_video_height,
    num_of_subimages,
    threshold_of_boundary,
    is_split_image2=True,
):

    # list for storing the new bboxes,classes and scores after reprojection
    new_bboxes = []
    new_classes = []
    new_scores = []

    # variables which store the index of the bboxes (in the list new_bboxes) which coincide with the left/right boundaries of the sub image
    left_boundary_box = None
    right_boundary_box = None

    # calculate the overlapped degree between each pair of the adjacent sub images (if the number of sub images is 4, then the results will be 30)
    overlaped_degree = (num_of_subimages * 120 - 360) / num_of_subimages
    # calculate which subimage will be splited into two parts (if the number of sub images is 4, then the results will be image 2)
    num_of_splited_subimage = num_of_subimages / 2

    index = 0
    # number of pixels occupied by (overlaped_degree/2) degrees on the sub image
    margin = int(lon_map_original.shape[0] / 120 * (overlaped_degree / 2))

    # for each bbox, class and score
    for bbox, class1, score in zip(bboxes, classes, scores):

        # get the coordinates of the top left point and the right bottom point
        left_top_x = int(bbox[0])
        left_top_y = int(bbox[1])
        right_bottom_x = int(bbox[2])
        right_bottom_y = int(bbox[3])

        # only reproject the bboxes when they are not totally inside the overlapped area and their y-values are less than 70 degrees (or sometimes the backpack of the cyclist will be incorrectly detected as a car)
        if (
            margin
            <= ((left_top_x + right_bottom_x) / 2)
            <= (lon_map_original.shape[0] - margin)
            and left_top_y <= lon_map_original.shape[0] / 120 * 70
        ):

            # since for an a*b sub image, the size of lon_map and lat_map is (a-1)*(b-1), when right_bottom_x or right_bottom_y equals a or b,
            # to get the corresponding value in lon_map and lat_map (which represent the corresponding position on the original image), we have to subtract them by 1.
            if right_bottom_x == lon_map_original.shape[0]:
                right_bottom_x -= 1
            if right_bottom_y == lon_map_original.shape[1]:
                right_bottom_y -= 1

            # check if a bbox coincides with the left/right boundaries of the sub image, if yes, assign its index to left_boundary_box/right_boundary_box
            # if the bbox is large (>subimage size/5), use the threshold to do the judgement
            if (right_bottom_x - left_top_x) * (
                right_bottom_y - left_top_y
            ) < lon_map_original.shape[0] * lon_map_original.shape[0] / 5:
                if left_top_x <= threshold_of_boundary:
                    left_boundary_box = index
                if right_bottom_x >= lon_map_original.shape[0] - threshold_of_boundary:
                    right_boundary_box = index

            # if the bbox is small (<=subimage size/5), set the threshold a little bit larger
            # (No why, just based on my experience ^_^)
            else:
                if left_top_x <= (
                    threshold_of_boundary + 15 * int(lon_map_original.shape[0] / 640)
                ):
                    left_boundary_box = index
                if right_bottom_x >= lon_map_original.shape[0] - (
                    threshold_of_boundary + 15 * int(lon_map_original.shape[0] / 640)
                ):
                    right_boundary_box = index

            # lists used to store the corresponding x and y coordinates on the original image of each point on the bbox
            xs = []
            ys = []

            # if the current sub image is the one which crosses the boundary (e.g., image 2 when the number of sub image is 4)
            # and the current bbox is across the center line
            if (
                num_of_subimage == num_of_splited_subimage
                and left_top_x <= int(lon_map_original.shape[0] / 2) - 1
                and right_bottom_x >= int(lon_map_original.shape[0] / 2)
            ):
                # lists used to store the x coordinates on the original image of each point on the left/right part of the bbox
                xs_left = []
                xs_right = []

                # calculation for the left and right borders
                for i in range(left_top_y, right_bottom_y, interval):
                    # left border
                    x = int(round(lon_map_original[i, left_top_x]))
                    y = int(round(lat_map_original[i, left_top_x]))
                    xs.append(x)
                    ys.append(y)
                    xs_left.append(x)
                    # right border
                    x = int(round(lon_map_original[i, right_bottom_x]))
                    y = int(round(lat_map_original[i, right_bottom_x]))
                    xs.append(x)
                    ys.append(y)
                    xs_right.append(x)

                # calculation for the left part of the top and bottom borders
                for i in range(
                    left_top_x, int(lon_map_original.shape[0] / 2) - 1, interval
                ):
                    x = int(round(lon_map_original[left_top_y, i]))
                    y = int(round(lat_map_original[left_top_y, i]))
                    xs.append(x)
                    ys.append(y)
                    xs_left.append(x)
                    x = int(round(lon_map_original[right_bottom_y, i]))
                    y = int(round(lat_map_original[right_bottom_y, i]))
                    xs.append(x)
                    ys.append(y)
                    xs_left.append(x)

                # calculation for the right part of the top and bottom borders
                for i in range(
                    int(lon_map_original.shape[0] / 2), right_bottom_x, interval
                ):
                    x = int(round(lon_map_original[left_top_y, i]))
                    y = int(round(lat_map_original[left_top_y, i]))
                    xs.append(x)
                    ys.append(y)
                    xs_right.append(x)
                    x = int(round(lon_map_original[right_bottom_y, i]))
                    y = int(round(lat_map_original[right_bottom_y, i]))
                    xs.append(x)
                    ys.append(y)
                    xs_right.append(x)

                ymax = max(ys)
                ymin = min(ys)
                xmin_left = min(xs_left)
                xmax_right = max(xs_right)

                # if it is needed to split the bbox into two parts, create two bboxes with the MBRs of the left and right part seperately
                if is_split_image2 == True:
                    new_bboxes.append([xmin_left, ymin, input_video_width, ymax])
                    new_bboxes.append([0, ymin, xmax_right, ymax])
                    new_classes.append(int(class1))
                    new_classes.append(int(class1))
                    new_scores.append(score)
                    new_scores.append(score)
                    index += 2

                # if not, create one bbox which extends outside the right boundary
                else:
                    new_bboxes.append(
                        [xmin_left, ymin, input_video_width + xmax_right, ymax]
                    )
                    new_classes.append(int(class1))
                    new_scores.append(score)
                    index += 1

            # if the current sub image is not the one which crosses the boundary
            else:
                # in case the interval is set larger than the length of the border, if so, set it as the length of the short side of the bbox
                if (
                    right_bottom_x - left_top_x < interval
                    or right_bottom_y - left_top_y < interval
                ):
                    interval = min(
                        right_bottom_x - left_top_x, right_bottom_y - left_top_y
                    )

                # get the corresponding coordinates on the original image of each point on the boundary
                for i in range(left_top_y, right_bottom_y, interval):
                    x = int(round(lon_map_original[i, left_top_x]))
                    y = int(round(lat_map_original[i, left_top_x]))
                    xs.append(x)
                    ys.append(y)
                    x = int(round(lon_map_original[i, right_bottom_x]))
                    y = int(round(lat_map_original[i, right_bottom_x]))
                    xs.append(x)
                    ys.append(y)
                for i in range(left_top_x, right_bottom_x, interval):
                    x = int(round(lon_map_original[left_top_y, i]))
                    y = int(round(lat_map_original[left_top_y, i]))
                    xs.append(x)
                    ys.append(y)
                    x = int(round(lon_map_original[right_bottom_y, i]))
                    y = int(round(lat_map_original[right_bottom_y, i]))
                    xs.append(x)
                    ys.append(y)

                # create one bbox with the MBR
                xmax = max(xs)
                xmin = min(xs)
                ymax = max(ys)
                ymin = min(ys)
                new_bboxes.append([xmin, ymin, xmax, ymax])
                new_classes.append(int(class1))
                new_scores.append(score)
                index += 1

    return new_bboxes, new_classes, new_scores, left_boundary_box, right_boundary_box


# function used to match the serial number of the sub image with the serial number of boundary
def number_of_left_and_right_boundary(number_of_subimage):
    if number_of_subimage == 0:
        return [2, 5]
    elif number_of_subimage == 1:
        return [4, 7]
    elif number_of_subimage == 2:
        return [6, 1]
    else:
        return [0, 3]


# function used to merge the bounding boxes of the objects which are shown in several sub images
def merge_bbox_across_boundary(
    bboxes_all, classes_all, scores_all, width, height, bboxes_boundary
):

    # list to store the index of the bbox to be deleted after we merge them
    bboxes_to_delete = []

    # first delete the bboxes which are on the boundary and are totally in the overlapped areas
    names = locals()
    for i in range(0, 8, 1):
        if bboxes_boundary[i] != None:
            #  although the overlapped area is 30 degree in width, here we set the threshold as 40, for after some tests, it seems 40 can get better performance.
            if (
                bboxes_all[bboxes_boundary[i]][2] - bboxes_all[bboxes_boundary[i]][0]
            ) <= int(width / 360 * 40):
                bboxes_to_delete.append(bboxes_boundary[i])
                bboxes_boundary[i] = None

    # Assign each value in the array to 8 variables, just for better understanding
    bboxes_boundary1 = bboxes_boundary[0]
    bboxes_boundary2 = bboxes_boundary[1]
    bboxes_boundary3 = bboxes_boundary[2]
    bboxes_boundary4 = bboxes_boundary[3]
    bboxes_boundary5 = bboxes_boundary[4]
    bboxes_boundary6 = bboxes_boundary[5]
    bboxes_boundary7 = bboxes_boundary[6]
    bboxes_boundary8 = bboxes_boundary[7]

    # if the object crosses all the 4 overlapped areas (12 34 56 78)
    if (
        bboxes_boundary1 != None
        and bboxes_boundary2 != None
        and bboxes_boundary3 != None
        and bboxes_boundary4 != None
        and bboxes_boundary5 != None
        and bboxes_boundary6 != None
        and bboxes_boundary7 != None
        and bboxes_boundary8 != None
        and (bboxes_boundary1 == bboxes_boundary4)
        and (bboxes_boundary3 == bboxes_boundary6)
        and (bboxes_boundary5 == bboxes_boundary8)
    ):
        bboxes_all.extend(
            MBR_bboxes(
                [
                    bboxes_all[bboxes_boundary2],
                    bboxes_all[bboxes_boundary1],
                    bboxes_all[bboxes_boundary3],
                    bboxes_all[bboxes_boundary5],
                    bboxes_all[bboxes_boundary7],
                ]
            )
        )
        classes_all.append(
            class_with_largest_score(
                [
                    bboxes_all[bboxes_boundary2],
                    bboxes_all[bboxes_boundary1],
                    bboxes_all[bboxes_boundary3],
                    bboxes_all[bboxes_boundary5],
                    bboxes_all[bboxes_boundary7],
                ],
                [
                    scores_all[bboxes_boundary2],
                    scores_all[bboxes_boundary1],
                    scores_all[bboxes_boundary3],
                    scores_all[bboxes_boundary5],
                    scores_all[bboxes_boundary7],
                ],
                [
                    classes_all[bboxes_boundary2],
                    classes_all[bboxes_boundary1],
                    classes_all[bboxes_boundary3],
                    classes_all[bboxes_boundary5],
                    classes_all[bboxes_boundary7],
                ],
            )
        )
        scores_all.extend(
            [
                weighted_average_score(
                    [
                        bboxes_all[bboxes_boundary2],
                        bboxes_all[bboxes_boundary1],
                        bboxes_all[bboxes_boundary3],
                        bboxes_all[bboxes_boundary5],
                        bboxes_all[bboxes_boundary7],
                    ],
                    [
                        scores_all[bboxes_boundary2],
                        scores_all[bboxes_boundary1],
                        scores_all[bboxes_boundary3],
                        scores_all[bboxes_boundary5],
                        scores_all[bboxes_boundary7],
                    ],
                )
            ]
        )
        bboxes_to_delete.extend(
            [
                bboxes_boundary1,
                bboxes_boundary2,
                bboxes_boundary3,
                bboxes_boundary4,
                bboxes_boundary5,
                bboxes_boundary6,
                bboxes_boundary7,
                bboxes_boundary8,
            ]
        )
    else:
        # if the object crosses 3 overlapped areas (12 34 56)
        if (
            bboxes_boundary1 != None
            and bboxes_boundary2 != None
            and bboxes_boundary3 != None
            and bboxes_boundary4 != None
            and bboxes_boundary5 != None
            and bboxes_boundary6 != None
            and (bboxes_boundary1 == bboxes_boundary4)
            and (bboxes_boundary3 == bboxes_boundary6)
        ):
            bboxes_all.extend(
                MBR_bboxes(
                    [
                        bboxes_all[bboxes_boundary2],
                        bboxes_all[bboxes_boundary1],
                        bboxes_all[bboxes_boundary3],
                        bboxes_all[bboxes_boundary5],
                    ]
                )
            )
            classes_all.append(
                class_with_largest_score(
                    [
                        bboxes_all[bboxes_boundary2],
                        bboxes_all[bboxes_boundary1],
                        bboxes_all[bboxes_boundary3],
                        bboxes_all[bboxes_boundary5],
                    ],
                    [
                        scores_all[bboxes_boundary2],
                        scores_all[bboxes_boundary1],
                        scores_all[bboxes_boundary3],
                        scores_all[bboxes_boundary5],
                    ],
                    [
                        classes_all[bboxes_boundary2],
                        classes_all[bboxes_boundary1],
                        classes_all[bboxes_boundary3],
                        classes_all[bboxes_boundary5],
                    ],
                )
            )
            scores_all.extend(
                [
                    weighted_average_score(
                        [
                            bboxes_all[bboxes_boundary2],
                            bboxes_all[bboxes_boundary1],
                            bboxes_all[bboxes_boundary3],
                            bboxes_all[bboxes_boundary5],
                        ],
                        [
                            scores_all[bboxes_boundary2],
                            scores_all[bboxes_boundary1],
                            scores_all[bboxes_boundary3],
                            scores_all[bboxes_boundary5],
                        ],
                    )
                ]
            )
            bboxes_to_delete.extend(
                [
                    bboxes_boundary1,
                    bboxes_boundary2,
                    bboxes_boundary3,
                    bboxes_boundary4,
                    bboxes_boundary5,
                    bboxes_boundary6,
                ]
            )

            # if another object crosses the remaining overlapped area (78)
            if bboxes_boundary7 != None and bboxes_boundary8 != None:
                bboxes_all.extend(
                    MBR_bboxes(
                        [bboxes_all[bboxes_boundary7], bboxes_all[bboxes_boundary8]]
                    )
                )
                classes_all.append(
                    class_with_largest_score(
                        [bboxes_all[bboxes_boundary8], bboxes_all[bboxes_boundary7]],
                        [scores_all[bboxes_boundary8], scores_all[bboxes_boundary7]],
                        [classes_all[bboxes_boundary8], classes_all[bboxes_boundary7]],
                    )
                )
                scores_all.extend(
                    [
                        weighted_average_score(
                            [
                                bboxes_all[bboxes_boundary8],
                                bboxes_all[bboxes_boundary7],
                            ],
                            [
                                scores_all[bboxes_boundary8],
                                scores_all[bboxes_boundary7],
                            ],
                        )
                    ]
                )
                bboxes_to_delete.extend([bboxes_boundary7, bboxes_boundary8])

        # if the object crosses 3 overlapped areas (34 56 78)
        if (
            bboxes_boundary3 != None
            and bboxes_boundary4 != None
            and bboxes_boundary5 != None
            and bboxes_boundary6 != None
            and bboxes_boundary7 != None
            and bboxes_boundary8 != None
            and (bboxes_boundary3 == bboxes_boundary6)
            and (bboxes_boundary5 == bboxes_boundary8)
        ):
            bboxes_all.extend(
                MBR_bboxes(
                    [
                        bboxes_all[bboxes_boundary4],
                        bboxes_all[bboxes_boundary3],
                        bboxes_all[bboxes_boundary5],
                        bboxes_all[bboxes_boundary7],
                    ]
                )
            )
            classes_all.append(
                class_with_largest_score(
                    [
                        bboxes_all[bboxes_boundary4],
                        bboxes_all[bboxes_boundary3],
                        bboxes_all[bboxes_boundary5],
                        bboxes_all[bboxes_boundary7],
                    ],
                    [
                        scores_all[bboxes_boundary4],
                        scores_all[bboxes_boundary3],
                        scores_all[bboxes_boundary5],
                        scores_all[bboxes_boundary7],
                    ],
                    [
                        classes_all[bboxes_boundary4],
                        classes_all[bboxes_boundary3],
                        classes_all[bboxes_boundary5],
                        classes_all[bboxes_boundary7],
                    ],
                )
            )
            scores_all.extend(
                [
                    weighted_average_score(
                        [
                            bboxes_all[bboxes_boundary4],
                            bboxes_all[bboxes_boundary3],
                            bboxes_all[bboxes_boundary5],
                            bboxes_all[bboxes_boundary7],
                        ],
                        [
                            scores_all[bboxes_boundary4],
                            scores_all[bboxes_boundary3],
                            scores_all[bboxes_boundary5],
                            scores_all[bboxes_boundary7],
                        ],
                    )
                ]
            )
            bboxes_to_delete.extend(
                [
                    bboxes_boundary3,
                    bboxes_boundary4,
                    bboxes_boundary5,
                    bboxes_boundary6,
                    bboxes_boundary7,
                    bboxes_boundary8,
                ]
            )

            # if another object crosses the remaining overlapped area (12)
            if bboxes_boundary1 != None and bboxes_boundary2 != None:
                bboxes_all.extend(
                    MBR_bboxes(
                        [bboxes_all[bboxes_boundary2], bboxes_all[bboxes_boundary1]]
                    )
                )
                classes_all.append(
                    class_with_largest_score(
                        [bboxes_all[bboxes_boundary2], bboxes_all[bboxes_boundary1]],
                        [scores_all[bboxes_boundary2], scores_all[bboxes_boundary1]],
                        [classes_all[bboxes_boundary2], classes_all[bboxes_boundary1]],
                    )
                )
                scores_all.extend(
                    [
                        weighted_average_score(
                            [
                                bboxes_all[bboxes_boundary2],
                                bboxes_all[bboxes_boundary1],
                            ],
                            [
                                scores_all[bboxes_boundary2],
                                scores_all[bboxes_boundary1],
                            ],
                        )
                    ]
                )
                bboxes_to_delete.extend([bboxes_boundary1, bboxes_boundary2])

        else:
            # if the object crosses 2 overlapped areas (12 34)
            if (
                bboxes_boundary1 != None
                and bboxes_boundary2 != None
                and bboxes_boundary3 != None
                and bboxes_boundary4 != None
                and (bboxes_boundary1 == bboxes_boundary4)
            ):
                bboxes_all.extend(
                    MBR_bboxes(
                        [
                            bboxes_all[bboxes_boundary2],
                            bboxes_all[bboxes_boundary1],
                            bboxes_all[bboxes_boundary3],
                        ]
                    )
                )
                classes_all.append(
                    class_with_largest_score(
                        [
                            bboxes_all[bboxes_boundary2],
                            bboxes_all[bboxes_boundary1],
                            bboxes_all[bboxes_boundary3],
                        ],
                        [
                            scores_all[bboxes_boundary2],
                            scores_all[bboxes_boundary1],
                            scores_all[bboxes_boundary3],
                        ],
                        [
                            classes_all[bboxes_boundary2],
                            classes_all[bboxes_boundary1],
                            classes_all[bboxes_boundary3],
                        ],
                    )
                )
                scores_all.extend(
                    [
                        weighted_average_score(
                            [
                                bboxes_all[bboxes_boundary2],
                                bboxes_all[bboxes_boundary1],
                                bboxes_all[bboxes_boundary3],
                            ],
                            [
                                scores_all[bboxes_boundary2],
                                scores_all[bboxes_boundary1],
                                scores_all[bboxes_boundary3],
                            ],
                        )
                    ]
                )
                bboxes_to_delete.extend(
                    [
                        bboxes_boundary1,
                        bboxes_boundary2,
                        bboxes_boundary3,
                        bboxes_boundary4,
                    ]
                )

                # if another object crosses the remaining overlapped area (56)
                if bboxes_boundary5 != None and bboxes_boundary6 != None:
                    bboxes_all.extend(
                        MBR_bboxes(
                            [bboxes_all[bboxes_boundary5], bboxes_all[bboxes_boundary6]]
                        )
                    )
                    classes_all.append(
                        class_with_largest_score(
                            [
                                bboxes_all[bboxes_boundary6],
                                bboxes_all[bboxes_boundary5],
                            ],
                            [
                                scores_all[bboxes_boundary6],
                                scores_all[bboxes_boundary5],
                            ],
                            [
                                classes_all[bboxes_boundary6],
                                classes_all[bboxes_boundary5],
                            ],
                        )
                    )
                    scores_all.extend(
                        [
                            weighted_average_score(
                                [
                                    bboxes_all[bboxes_boundary6],
                                    bboxes_all[bboxes_boundary5],
                                ],
                                [
                                    scores_all[bboxes_boundary6],
                                    scores_all[bboxes_boundary5],
                                ],
                            )
                        ]
                    )
                    bboxes_to_delete.extend([bboxes_boundary5, bboxes_boundary6])

                # if another object crosses the remaining overlapped area (78)
                if bboxes_boundary7 != None and bboxes_boundary8 != None:
                    bboxes_all.extend(
                        MBR_bboxes(
                            [bboxes_all[bboxes_boundary7], bboxes_all[bboxes_boundary8]]
                        )
                    )
                    classes_all.append(
                        class_with_largest_score(
                            [
                                bboxes_all[bboxes_boundary8],
                                bboxes_all[bboxes_boundary7],
                            ],
                            [
                                scores_all[bboxes_boundary8],
                                scores_all[bboxes_boundary7],
                            ],
                            [
                                classes_all[bboxes_boundary8],
                                classes_all[bboxes_boundary7],
                            ],
                        )
                    )
                    scores_all.extend(
                        [
                            weighted_average_score(
                                [
                                    bboxes_all[bboxes_boundary8],
                                    bboxes_all[bboxes_boundary7],
                                ],
                                [
                                    scores_all[bboxes_boundary8],
                                    scores_all[bboxes_boundary7],
                                ],
                            )
                        ]
                    )
                    bboxes_to_delete.extend([bboxes_boundary7, bboxes_boundary8])

            # if the object crosses 2 overlapped areas (34 56)
            if (
                bboxes_boundary3 != None
                and bboxes_boundary4 != None
                and bboxes_boundary5 != None
                and bboxes_boundary6 != None
                and (bboxes_boundary3 == bboxes_boundary6)
            ):
                bboxes_all.extend(
                    MBR_bboxes(
                        [
                            bboxes_all[bboxes_boundary4],
                            bboxes_all[bboxes_boundary3],
                            bboxes_all[bboxes_boundary5],
                        ]
                    )
                )
                classes_all.append(
                    class_with_largest_score(
                        [
                            bboxes_all[bboxes_boundary4],
                            bboxes_all[bboxes_boundary3],
                            bboxes_all[bboxes_boundary5],
                        ],
                        [
                            scores_all[bboxes_boundary4],
                            scores_all[bboxes_boundary3],
                            scores_all[bboxes_boundary5],
                        ],
                        [
                            classes_all[bboxes_boundary4],
                            classes_all[bboxes_boundary3],
                            classes_all[bboxes_boundary5],
                        ],
                    )
                )
                scores_all.extend(
                    [
                        weighted_average_score(
                            [
                                bboxes_all[bboxes_boundary4],
                                bboxes_all[bboxes_boundary3],
                                bboxes_all[bboxes_boundary5],
                            ],
                            [
                                scores_all[bboxes_boundary4],
                                scores_all[bboxes_boundary3],
                                scores_all[bboxes_boundary5],
                            ],
                        )
                    ]
                )
                bboxes_to_delete.extend(
                    [
                        bboxes_boundary3,
                        bboxes_boundary4,
                        bboxes_boundary5,
                        bboxes_boundary6,
                    ]
                )

                # if another object crosses the remaining overlapped area (12)
                if bboxes_boundary1 != None and bboxes_boundary2 != None:
                    bboxes_all.extend(
                        MBR_bboxes(
                            [bboxes_all[bboxes_boundary2], bboxes_all[bboxes_boundary1]]
                        )
                    )
                    classes_all.append(
                        class_with_largest_score(
                            [
                                bboxes_all[bboxes_boundary2],
                                bboxes_all[bboxes_boundary1],
                            ],
                            [
                                scores_all[bboxes_boundary2],
                                scores_all[bboxes_boundary1],
                            ],
                            [
                                classes_all[bboxes_boundary2],
                                classes_all[bboxes_boundary1],
                            ],
                        )
                    )
                    scores_all.extend(
                        [
                            weighted_average_score(
                                [
                                    bboxes_all[bboxes_boundary2],
                                    bboxes_all[bboxes_boundary1],
                                ],
                                [
                                    scores_all[bboxes_boundary2],
                                    scores_all[bboxes_boundary1],
                                ],
                            )
                        ]
                    )
                    bboxes_to_delete.extend([bboxes_boundary1, bboxes_boundary2])

                # if another object crosses the remaining overlapped area (78)
                if bboxes_boundary7 != None and bboxes_boundary8 != None:
                    bboxes_all.extend(
                        MBR_bboxes(
                            [bboxes_all[bboxes_boundary7], bboxes_all[bboxes_boundary8]]
                        )
                    )
                    classes_all.append(
                        class_with_largest_score(
                            [
                                bboxes_all[bboxes_boundary8],
                                bboxes_all[bboxes_boundary7],
                            ],
                            [
                                scores_all[bboxes_boundary8],
                                scores_all[bboxes_boundary7],
                            ],
                            [
                                classes_all[bboxes_boundary8],
                                classes_all[bboxes_boundary7],
                            ],
                        )
                    )
                    scores_all.extend(
                        [
                            weighted_average_score(
                                [
                                    bboxes_all[bboxes_boundary8],
                                    bboxes_all[bboxes_boundary7],
                                ],
                                [
                                    scores_all[bboxes_boundary8],
                                    scores_all[bboxes_boundary7],
                                ],
                            )
                        ]
                    )
                    bboxes_to_delete.extend([bboxes_boundary7, bboxes_boundary8])

            # if the object crosses 2 overlapped areas (56 78)
            if (
                bboxes_boundary5 != None
                and bboxes_boundary6 != None
                and bboxes_boundary7 != None
                and bboxes_boundary8 != None
                and (bboxes_boundary5 == bboxes_boundary8)
            ):
                bboxes_all.extend(
                    MBR_bboxes(
                        [
                            bboxes_all[bboxes_boundary6],
                            bboxes_all[bboxes_boundary5],
                            bboxes_all[bboxes_boundary7],
                        ]
                    )
                )
                classes_all.append(
                    class_with_largest_score(
                        [
                            bboxes_all[bboxes_boundary6],
                            bboxes_all[bboxes_boundary5],
                            bboxes_all[bboxes_boundary7],
                        ],
                        [
                            scores_all[bboxes_boundary6],
                            scores_all[bboxes_boundary5],
                            scores_all[bboxes_boundary7],
                        ],
                        [
                            classes_all[bboxes_boundary6],
                            classes_all[bboxes_boundary5],
                            classes_all[bboxes_boundary7],
                        ],
                    )
                )
                scores_all.extend(
                    [
                        weighted_average_score(
                            [
                                bboxes_all[bboxes_boundary6],
                                bboxes_all[bboxes_boundary5],
                                bboxes_all[bboxes_boundary7],
                            ],
                            [
                                scores_all[bboxes_boundary6],
                                scores_all[bboxes_boundary5],
                                scores_all[bboxes_boundary7],
                            ],
                        )
                    ]
                )
                bboxes_to_delete.extend(
                    [
                        bboxes_boundary5,
                        bboxes_boundary6,
                        bboxes_boundary7,
                        bboxes_boundary8,
                    ]
                )

                # if another object crosses the remaining overlapped area (12)
                if bboxes_boundary1 != None and bboxes_boundary2 != None:
                    bboxes_all.extend(
                        MBR_bboxes(
                            [bboxes_all[bboxes_boundary2], bboxes_all[bboxes_boundary1]]
                        )
                    )
                    classes_all.append(
                        class_with_largest_score(
                            [
                                bboxes_all[bboxes_boundary2],
                                bboxes_all[bboxes_boundary1],
                            ],
                            [
                                scores_all[bboxes_boundary2],
                                scores_all[bboxes_boundary1],
                            ],
                            [
                                classes_all[bboxes_boundary2],
                                classes_all[bboxes_boundary1],
                            ],
                        )
                    )
                    scores_all.extend(
                        [
                            weighted_average_score(
                                [
                                    bboxes_all[bboxes_boundary2],
                                    bboxes_all[bboxes_boundary1],
                                ],
                                [
                                    scores_all[bboxes_boundary2],
                                    scores_all[bboxes_boundary1],
                                ],
                            )
                        ]
                    )
                    bboxes_to_delete.extend([bboxes_boundary1, bboxes_boundary2])

                # if another object crosses the remaining overlapped area (34)
                if bboxes_boundary3 != None and bboxes_boundary4 != None:
                    bboxes_all.extend(
                        MBR_bboxes(
                            [bboxes_all[bboxes_boundary3], bboxes_all[bboxes_boundary4]]
                        )
                    )
                    classes_all.append(
                        class_with_largest_score(
                            [
                                bboxes_all[bboxes_boundary4],
                                bboxes_all[bboxes_boundary3],
                            ],
                            [
                                scores_all[bboxes_boundary4],
                                scores_all[bboxes_boundary3],
                            ],
                            [
                                classes_all[bboxes_boundary4],
                                classes_all[bboxes_boundary3],
                            ],
                        )
                    )
                    scores_all.extend(
                        [
                            weighted_average_score(
                                [
                                    bboxes_all[bboxes_boundary4],
                                    bboxes_all[bboxes_boundary3],
                                ],
                                [
                                    scores_all[bboxes_boundary4],
                                    scores_all[bboxes_boundary3],
                                ],
                            )
                        ]
                    )
                    bboxes_to_delete.extend([bboxes_boundary3, bboxes_boundary4])

            else:
                # if the object crosses 1 overlapped area (12)
                if bboxes_boundary1 != None and bboxes_boundary2 != None:
                    bboxes_all.extend(
                        MBR_bboxes(
                            [bboxes_all[bboxes_boundary2], bboxes_all[bboxes_boundary1]]
                        )
                    )
                    classes_all.append(
                        class_with_largest_score(
                            [
                                bboxes_all[bboxes_boundary2],
                                bboxes_all[bboxes_boundary1],
                            ],
                            [
                                scores_all[bboxes_boundary2],
                                scores_all[bboxes_boundary1],
                            ],
                            [
                                classes_all[bboxes_boundary2],
                                classes_all[bboxes_boundary1],
                            ],
                        )
                    )
                    scores_all.extend(
                        [
                            weighted_average_score(
                                [
                                    bboxes_all[bboxes_boundary2],
                                    bboxes_all[bboxes_boundary1],
                                ],
                                [
                                    scores_all[bboxes_boundary2],
                                    scores_all[bboxes_boundary1],
                                ],
                            )
                        ]
                    )
                    bboxes_to_delete.extend([bboxes_boundary1, bboxes_boundary2])

                # if the object crosses 1 overlapped area (34)
                if bboxes_boundary3 != None and bboxes_boundary4 != None:
                    bboxes_all.extend(
                        MBR_bboxes(
                            [bboxes_all[bboxes_boundary3], bboxes_all[bboxes_boundary4]]
                        )
                    )
                    classes_all.append(
                        class_with_largest_score(
                            [
                                bboxes_all[bboxes_boundary4],
                                bboxes_all[bboxes_boundary3],
                            ],
                            [
                                scores_all[bboxes_boundary4],
                                scores_all[bboxes_boundary3],
                            ],
                            [
                                classes_all[bboxes_boundary4],
                                classes_all[bboxes_boundary3],
                            ],
                        )
                    )
                    scores_all.extend(
                        [
                            weighted_average_score(
                                [
                                    bboxes_all[bboxes_boundary4],
                                    bboxes_all[bboxes_boundary3],
                                ],
                                [
                                    scores_all[bboxes_boundary4],
                                    scores_all[bboxes_boundary3],
                                ],
                            )
                        ]
                    )
                    bboxes_to_delete.extend([bboxes_boundary3, bboxes_boundary4])

                # if the object crosses 1 overlapped area (56)
                if bboxes_boundary5 != None and bboxes_boundary6 != None:
                    bboxes_all.extend(
                        MBR_bboxes(
                            [bboxes_all[bboxes_boundary5], bboxes_all[bboxes_boundary6]]
                        )
                    )
                    classes_all.append(
                        class_with_largest_score(
                            [
                                bboxes_all[bboxes_boundary6],
                                bboxes_all[bboxes_boundary5],
                            ],
                            [
                                scores_all[bboxes_boundary6],
                                scores_all[bboxes_boundary5],
                            ],
                            [
                                classes_all[bboxes_boundary6],
                                classes_all[bboxes_boundary5],
                            ],
                        )
                    )
                    scores_all.extend(
                        [
                            weighted_average_score(
                                [
                                    bboxes_all[bboxes_boundary6],
                                    bboxes_all[bboxes_boundary5],
                                ],
                                [
                                    scores_all[bboxes_boundary6],
                                    scores_all[bboxes_boundary5],
                                ],
                            )
                        ]
                    )
                    bboxes_to_delete.extend([bboxes_boundary5, bboxes_boundary6])

                # if the object crosses 1 overlapped area (78)
                if bboxes_boundary7 != None and bboxes_boundary8 != None:
                    bboxes_all.extend(
                        MBR_bboxes(
                            [bboxes_all[bboxes_boundary7], bboxes_all[bboxes_boundary8]]
                        )
                    )
                    classes_all.append(
                        class_with_largest_score(
                            [
                                bboxes_all[bboxes_boundary8],
                                bboxes_all[bboxes_boundary7],
                            ],
                            [
                                scores_all[bboxes_boundary8],
                                scores_all[bboxes_boundary7],
                            ],
                            [
                                classes_all[bboxes_boundary8],
                                classes_all[bboxes_boundary7],
                            ],
                        )
                    )
                    scores_all.extend(
                        [
                            weighted_average_score(
                                [
                                    bboxes_all[bboxes_boundary8],
                                    bboxes_all[bboxes_boundary7],
                                ],
                                [
                                    scores_all[bboxes_boundary8],
                                    scores_all[bboxes_boundary7],
                                ],
                            )
                        ]
                    )
                    bboxes_to_delete.extend([bboxes_boundary7, bboxes_boundary8])

    # delete the boxes that have been merged from the lists
    bboxes_to_delete = list(set(bboxes_to_delete))
    bboxes_to_delete.sort(reverse=True)
    for i in bboxes_to_delete:
        bboxes_all.pop(i)
        classes_all.pop(i)
        scores_all.pop(i)

    return bboxes_all, classes_all, scores_all


# function used to calculate the weighted average score of several bboxes
def weighted_average_score(bboxes, scores):
    sum = 0
    sum_area = 0
    for bbox, score in zip(bboxes, scores):
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        sum += score * area
        sum_area += area
    return np.float32(sum / sum_area)


# function used to choose the class with the largest weighted score as the class of the new merged bbox
def class_with_largest_score(bboxes, scores, classes):
    sum_area = 0
    score_multi_area = []
    for bbox, score in zip(bboxes, scores):
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        score_multi_area.append(area * score)
        sum_area += area
    weighted_score = [i / sum_area for i in score_multi_area]
    return classes[weighted_score.index(max(weighted_score))]


# function used to calculate the MBR of several connected bboxes
def MBR_bboxes(bboxes):
    xs = []
    ys = []
    for bbox in bboxes:
        xs.append(bbox[0])
        xs.append(bbox[2])
        ys.append(bbox[1])
        ys.append(bbox[3])
    return [[min(xs), min(ys), max(xs), max(ys)]]


# function used to filter the bboxes according to the classes we need
def filter_classes(bboxes_all, classes_all, scores_all, class_needed):
    bboxes_all = bboxes_all.tolist()
    classes_all = classes_all.tolist()
    scores_all = scores_all.tolist()
    # remove the bboxes which are not belong to the needed classes from the lists
    for i in range(len(classes_all), 0, -1):
        if classes_all[i - 1] not in class_needed:
            bboxes_all.pop(i - 1)
            classes_all.pop(i - 1)
            scores_all.pop(i - 1)
    return bboxes_all, classes_all, scores_all


# function used to project the class id from [0,1,2,3,5,7,9] to [0,6] to match our annotations
def project_class(classes):
    for index, class1 in enumerate(classes):
        if class1 == 5:
            classes[index] = 4
        elif class1 == 7:
            classes[index] = 5
        elif class1 == 9:
            classes[index] = 6
    return classes


# A function used to transform the output from [x1,y1,x2,y2] format to [x_centre, y_centre, width, height].
def xyxy2xcycwh(bboxes):
    bboxes_new = []
    for bbox in bboxes:
        bboxes_new.append(
            [
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2,
                (bbox[2] - bbox[0]),
                (bbox[3] - bbox[1]),
            ]
        )
    return bboxes_new


# function used to do object detection on one image frame
def predict_one_frame(
    FOV,
    THETAs,
    PHIs,
    im,
    predictor,
    video_width,
    video_height,
    sub_image_width,
    classes_to_detect=[0, 1, 2, 3, 5, 7, 9],
    is_project_class=False,
    use_mymodel=True,
    model="Faster RCNN",
    split_image2=True,
):

    # for checking the processing speed, record the current time first
    time1 = time.time()

    # if the user chooses to use the improved object detection model
    if use_mymodel:
        # split the frame into 4 sub images (of perspective projection) and get the maps and the output images
        lon_maps, lat_maps, subimgs = equir2pers(
            im, FOV, THETAs, PHIs, sub_image_width, sub_image_width
        )

        # lists for storing the detection results from all the sub images
        bboxes_all = []
        classes_all = []
        scores_all = []

        # list for storing the index of the bounding boxes which intersect with the boundaries of the sub images
        bboxes_boundary = [None] * 8

        # if a Faster RCNN model is being used
        if model == "Faster RCNN":
            # for each sub image
            for i in range(len(subimgs)):
                # get the detection results with the predictor
                outputs1 = predictor(subimgs[i])

                # --------  if you want to save and check the detail of the results on each sub image, run the code below  ----------
                # v1 = Visualizer(
                #     subimgs[i][:, :, ::-1],
                #     MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                #     scale=1.0,
                # )
                # im1 = v1.draw_instance_predictions(outputs1["instances"].to("cpu"))
                # cv2.imwrite(
                #     "./outtest/subdetect" + str(i) + ".png", im1.get_image()[:, :, ::-1]
                # )
                # --------  end of this part  ----------

                # get the bboxes, classes and scores of the instances detected
                bboxes = outputs1["instances"].pred_boxes.tensor.cpu().numpy()
                classes = outputs1["instances"].pred_classes.cpu().numpy()
                scores = outputs1["instances"].scores.cpu().numpy()

                # do NMS on the bboxes despite the category
                # keep_boxes is a list which stores the index of the bboxes to keep after NMS
                keep_boxes = torchvision.ops.nms(
                    torch.tensor(bboxes), torch.tensor(scores), 0.45
                )

                # for each bbox in the current sub image, reproject it to the original image
                (
                    reprojected_bboxes,
                    classes,
                    scores,
                    left_boundary_box,
                    right_boundary_box,
                ) = reproject_bboxes(
                    torch.tensor(bboxes)[keep_boxes],
                    lon_maps[i],
                    lat_maps[i],
                    torch.tensor(classes)[keep_boxes],
                    torch.tensor(scores)[keep_boxes],
                    10,
                    i,
                    video_width,
                    video_height,
                    len(subimgs),
                    sub_image_width / 640 * 20,
                    split_image2,
                )

                # get the index of the bboxes which intersect the boundaries of the sub images
                if left_boundary_box != None:
                    bboxes_boundary[
                        number_of_left_and_right_boundary(i)[0]
                    ] = left_boundary_box + len(bboxes_all)
                if right_boundary_box != None:
                    bboxes_boundary[
                        number_of_left_and_right_boundary(i)[1]
                    ] = right_boundary_box + len(bboxes_all)

                # add the bboxes after reprojection to the lists which contain bboxes from all the sub images
                bboxes_all = bboxes_all + reprojected_bboxes
                classes_all = classes_all + classes
                scores_all = scores_all + scores

        # if a YOLO model is being used
        elif model == "YOLO":

            # for each sub image, first change the color from BGR to RGB
            for i in range(len(subimgs)):
                subimgs[i] = cv2.cvtColor(subimgs[i], cv2.COLOR_BGR2RGB)

            # YOLO supports detecting several images at the same time, so input all the sub images at once to the predictor
            results = predictor(subimgs, size=sub_image_width)  # includes NMS

            # --------  if you want to save and check the detail of the results on each sub image, run the code below  ----------
            # results.save()
            # --------  end of this part  ----------

            # for each sub image
            for i in range(len(subimgs)):
                # Originally, YOLO outputs the positions using the relative coordinates [0-1], so transform the output format by multiplying by the width/height of the sub image
                bboxes = (
                    results.xyxyn[i].cpu().numpy()[:, 0:4]
                    * [
                        sub_image_width,
                        sub_image_width,
                        sub_image_width,
                        sub_image_width,
                    ]
                ).tolist()
                classes = list(map(int, results.xyxyn[i].cpu().numpy()[:, 5].tolist()))
                scores = results.xyxyn[i].cpu().numpy()[:, 4].tolist()

                # for each bbox in the current sub image, reproject it to the original image
                (
                    reprojected_bboxes,
                    classes,
                    scores,
                    left_boundary_box,
                    right_boundary_box,
                ) = reproject_bboxes(
                    bboxes,
                    lon_maps[i],
                    lat_maps[i],
                    classes,
                    scores,
                    10,
                    i,
                    video_width,
                    video_height,
                    len(subimgs),
                    sub_image_width / 640 * 20,
                    split_image2,
                )

                # get the index of the bboxes which intersect the boundaries of the sub images
                if left_boundary_box != None:
                    bboxes_boundary[
                        number_of_left_and_right_boundary(i)[0]
                    ] = left_boundary_box + len(bboxes_all)
                if right_boundary_box != None:
                    bboxes_boundary[
                        number_of_left_and_right_boundary(i)[1]
                    ] = right_boundary_box + len(bboxes_all)

                # add the bboxes after reprojection to the lists which contain bboxes from all the sub images
                bboxes_all = bboxes_all + reprojected_bboxes
                classes_all = classes_all + classes
                scores_all = scores_all + scores

        # merge the boxes which goes across the boundaries with merge_bbox_across_boundary()
        bboxes_all, classes_all, scores_all = merge_bbox_across_boundary(
            bboxes_all,
            classes_all,
            scores_all,
            video_width,
            video_height,
            bboxes_boundary,
        )

        # do NMS on the output bboxes again to get the index of the boxes which should be kept
        keep = batched_nms(
            torch.tensor(bboxes_all),
            torch.tensor(scores_all),
            torch.tensor(classes_all),
            0.3,
        )

        # only keep the instances of the classes we need (person, bike, car, motorbike, bus, truck, traffic light by default)
        bboxes_all, classes_all, scores_all = filter_classes(
            torch.tensor(bboxes_all)[keep],
            torch.tensor(classes_all)[keep],
            torch.tensor(scores_all)[keep],
            classes_to_detect,
        )

        # if needed, project the class into [0,6] (to match with the annotations in our dataset)
        if is_project_class == True:
            classes_all = project_class(classes_all)

    # if the user chooses to use the original object detection model
    else:
        # if a Faster RCNN model is being used
        if model == "Faster RCNN":
            # get the outputs and do NMS on them
            outputs1 = predictor(im)
            bboxes_all = outputs1["instances"].pred_boxes.tensor.cpu().numpy()
            classes_all = outputs1["instances"].pred_classes.cpu().numpy()
            scores_all = outputs1["instances"].scores.cpu().numpy()
            keep_boxes = torchvision.ops.nms(
                torch.tensor(bboxes_all), torch.tensor(scores_all), 0.45
            )
            bboxes_all = (
                outputs1["instances"].pred_boxes.tensor.cpu().numpy()[keep_boxes]
            )
            classes_all = outputs1["instances"].pred_classes.cpu().numpy()[keep_boxes]
            scores_all = outputs1["instances"].scores.cpu().numpy()[keep_boxes]

        # if a YOLO model is being used
        elif model == "YOLO":
            # change the color from BGR to RGB
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # get the outputs
            results = predictor(im, size=sub_image_width)  # NMS included
            bboxes_all = (
                results.xyxyn[0].cpu().numpy()[:, 0:4]
                * [video_width, video_height, video_width, video_height]
            ).tolist()
            classes_all = list(map(int, results.xyxyn[0].cpu().numpy()[:, 5].tolist()))
            scores_all = results.xyxyn[0].cpu().numpy()[:, 4].tolist()

        # only keep the instances of the classes we need (person, bike, car, motorbike, bus, truck, traffic light)
        bboxes_all, classes_all, scores_all = filter_classes(
            torch.tensor(bboxes_all),
            torch.tensor(classes_all),
            torch.tensor(scores_all),
            classes_to_detect,
        )

        # if needed, project the class into [0,6] (to match with the annotations in our dataset)
        if is_project_class == True:
            classes_all = project_class(classes_all)

    # record the current time again and calculate the running time
    time2 = time.time()
    # print(time2 - time1)

    return bboxes_all, classes_all, scores_all
