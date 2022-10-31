from contextlib import redirect_stderr
import numpy as np
import cv2
import math

palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)

# function used to provide random colors for the bounding boxes
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label**2 - label + 1)) % 255) for p in palette]
    return tuple(color)


# function used to draw the tracking/overtake results on the video
def draw_boxes(
    img,
    bbox,
    track_classes,
    track_scores,
    video_width,
    identities=None,
    backwards_tracks=None,
    forwards_tracks=None,
    unconfirmed_left_overtaking=None,
    unconfirmed_right_overtaking=None,
    threshold=[500 * 500, 900 * 900, 600 * 600],
    close_overtaking_warning=False,
    classes_to_detect_movement=[2, 4, 5],
    offset=(0, 0),
):
    # if close overtaking warning is on
    if close_overtaking_warning == True:

        # draw a meter showing the direction of the cars which take over the cyclist
        white_area = np.zeros(img.shape, np.uint8)
        cv2.circle(white_area, (500, 500), 300, (255, 255, 255), thickness=-1)
        img = cv2.addWeighted(img, 1, white_area, 0.6, 1)
        cv2.circle(img, (500, 500), 300, (0, 0, 0), thickness=5)
        cv2.putText(
            img,
            "0",
            (500, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            [0, 0, 0],
            2,
        )

        cv2.putText(
            img,
            "90",
            (820, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            [0, 0, 0],
            2,
        )

        cv2.putText(
            img,
            "180",
            (480, 830),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            [0, 0, 0],
            2,
        )

        cv2.putText(
            img,
            "-90",
            (120, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            [0, 0, 0],
            2,
        )

    # for each object, plot the box and label of it
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        bbox_size = None
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)

        # if the object is moving backwards/forwards, show it in the label
        if backwards_tracks != None and id in backwards_tracks:
            label = (
                str(id)
                + " "
                + classid2name(track_classes[i])
                + " "
                + str(round(track_scores[i] * 100, 2))
                + "%"
                + " Backwards"
            )

        elif forwards_tracks != None and id in forwards_tracks:
            label = (
                str(id)
                + " "
                + classid2name(track_classes[i])
                + " "
                + str(round(track_scores[i] * 100, 2))
                + "%"
                + " Forwards"
            )

            if close_overtaking_warning == True and (
                id in unconfirmed_left_overtaking or id in unconfirmed_right_overtaking
            ):
                # if the size of the objects moving forwards is lager than a threshold
                bbox_size = (x2 - x1) * (y2 - y1)
                fillred = False
                for class1, threshold1 in zip(classes_to_detect_movement, threshold):
                    if track_classes[i] == class1 and bbox_size > threshold1:
                        fillred = True
                if fillred:
                    # fill the bounding box with red
                    red_area = np.zeros(img.shape, np.uint8)
                    cv2.rectangle(red_area, (x1, y1), (x2, y2), (0, 0, 255), -1)
                    img = cv2.addWeighted(img, 1.0, red_area, 0.5, 1)
                    # draw an arrow in the white metre which shows the direction of the dangerous object
                    angle = (((x1 + x2) / 2) / video_width * 2 * math.pi) - math.pi
                    # print(angle)
                    print(
                        int(500 + math.sin(angle) * 500),
                        int(500 - math.cos(angle) * 500),
                    )
                    cv2.arrowedLine(
                        img,
                        (500, 500),
                        (
                            int(500 + math.sin(angle) * 300),
                            int(500 - math.cos(angle) * 300),
                        ),
                        (0, 0, 255),
                        5,
                        0,
                        0,
                        0.1,
                    )

        else:
            label = (
                str(id)
                + " "
                + classid2name(track_classes[i])
                + " "
                + str(round(track_scores[i] * 100, 2))
                + "%"
            )

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]

        # if the bbox is totally in the image frame
        if x2 <= video_width:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 15, y1 + t_size[1] + 15), color, -1
            )
            cv2.putText(
                img,
                label,
                (x1, y1 + t_size[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                [255, 255, 255],
                2,
            )
        # if the bbox crosses the boundary of the video
        else:
            cv2.rectangle(img, (x1, y1), (video_width, y2), color, 3)
            # plot the right part
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 15, y1 + t_size[1] + 15), color, -1
            )
            cv2.putText(
                img,
                label,
                (x1, y1 + t_size[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                [255, 255, 255],
                2,
            )
            # plot the left part
            cv2.rectangle(img, (0, y1), (x2 - video_width, y2), color, 3)
            cv2.rectangle(
                img, (0, y1), (0 + t_size[0] + 15, y1 + t_size[1] + 15), color, -1
            )
            cv2.putText(
                img,
                label,
                (0, y1 + t_size[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                [255, 255, 255],
                2,
            )
    return img


# a function used to get the corresponding class of the index number in COCO dataset
def classid2name(id):
    names = [
        "person",
        "bicycle",
        "car",
        "motorbike",
        "aeroplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "sofa",
        "pottedplant",
        "bed",
        "diningtable",
        "toilet",
        "tvmonitor",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]
    return names[int(id)]


if __name__ == "__main__":
    for i in range(82):
        print(compute_color_for_labels(i))
