import argparse
import time
import torch
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
from panoramic_detection import improved_OD as OD

# function used to realize object detection on a panoramic video
def Object_Detection(
    input_video_path,
    output_video_path,
    classes_to_detect=[0, 1, 2, 3, 5, 7, 9],
    FOV=120,
    THETAs=[0, 90, 180, 270],
    PHIs=[-10, -10, -10, -10],
    sub_image_width=640,
    model_type="YOLO",
    score_threshold=0.4,
    nms_threshold=0.45,
    use_mymodel=True,
):

    print("Loading Model...")
    # load the pretrained detection model
    model, cfg = OD.load_model(
        model_type, sub_image_width, score_threshold, nms_threshold
    )
    print("Model Loaded!")

    # read the input panoramic video (of equirectangular projection)
    video_capture = cv2.VideoCapture(input_video_path)

    # if the input path is not right, warn the user
    if video_capture.isOpened() == False:
        print("Can not open the video file.")
    # if right, get some info about the video (width, height, frame count and fps)
    else:
        video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = int(round(video_capture.get(cv2.CAP_PROP_FPS)))
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        outputfile = cv2.VideoWriter(
            output_video_path, fourcc, video_fps, (video_width, video_height)
        )
    # print(video_frame_count)
    # output the video info
    print(
        "The input video is "
        + str(video_width)
        + " in width and "
        + str(video_height)
        + " in height."
    )

    # the number of current frame
    num_of_frame = 1

    # for each image frame in the video
    while video_capture.grab():

        time1 = time.time()

        # get the next image frame
        _, im = video_capture.retrieve()

        # get the predictions on the current frame
        bboxes_all, classes_all, scores_all = OD.predict_one_frame(
            FOV,
            THETAs,
            PHIs,
            im,
            model,
            video_width,
            video_height,
            sub_image_width,
            classes_to_detect,
            False,
            use_mymodel,
            model_type,
            True,
        )

        # create an instance of detectron2 so that the output can be visualized
        output_new = Instances(
            image_size=[video_width, video_height],
            pred_boxes=Boxes(torch.tensor(bboxes_all)),
            scores=torch.tensor(scores_all),
            pred_classes=torch.tensor(classes_all),
        )

        # use `Visualizer` to draw the predictions on the image
        v = Visualizer(
            im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0
        )
        im = v.draw_instance_predictions(output_new.to("cpu"))
        outputfile.write(im.get_image()[:, :, ::-1])

        # show the current FPS
        time2 = time.time()
        if num_of_frame % 5 == 0:
            print(num_of_frame, "/", video_frame_count)
            print(str(1 / (time2 - time1)) + " fps")

        num_of_frame += 1

    # release the input and output videos
    video_capture.release()
    outputfile.release()
    print("Output Finished!")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video_path", required=True, type=str)
    parser.add_argument("--output_video_path", required=True, type=str)
    parser.add_argument(
        "--classes_to_detect", nargs="+", type=int, default=[0, 1, 2, 3, 5, 7, 9]
    )
    parser.add_argument("--FOV", type=int, default=120)
    parser.add_argument("--THETAs", nargs="+", type=int, default=[0, 90, 180, 270])
    parser.add_argument("--PHIs", nargs="+", type=int, default=[-10, -10, -10, -10])
    parser.add_argument("--sub_image_width", type=int, default=640)
    parser.add_argument(
        "--model_type", type=str, choices=["YOLO", "Faster RCNN"], default="YOLO"
    )
    parser.add_argument("--score_threshold", type=float, default=0.4)
    parser.add_argument("--nms_threshold", type=float, default=0.45)
    parser.add_argument("--use_mymodel", default=True, type=boolean_string)
    opt = parser.parse_args()
    # print(opt)
    return opt


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def main(opt):
    Object_Detection(
        opt.input_video_path,
        opt.output_video_path,
        opt.classes_to_detect,
        opt.FOV,
        opt.THETAs,
        opt.PHIs,
        opt.sub_image_width,
        opt.model_type,
        opt.score_threshold,
        opt.nms_threshold,
        opt.use_mymodel,
    )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
