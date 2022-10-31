import argparse
import time
import torch
import cv2
import numpy as np

from panoramic_detection import improved_OD as OD
from deep_sort.deep_sort import DeepSort
from panoramic_detection.draw_output import draw_boxes

# a function used to realize overtaking behaviour detection on a panoramic video
def Overtaking_Detection(
    input_video_path,
    output_video_path,
    mode="Confirmed",
    prevent_different_classes_match=True,
    match_across_boundary=True,
    classes_to_detect=[0, 1, 2, 3, 5, 7, 9],
    classes_to_detect_movement=[2, 5, 7],
    size_thresholds=[500 * 500, 900 * 900, 600 * 600],
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

    # output the video info
    print(
        "The input video is "
        + str(video_width)
        + " in width and "
        + str(video_height)
        + " in height."
    )

    # create a deepsort instance with the pre-trained feature extraction model
    deepsort = DeepSort(
        "./deep_sort/deep/checkpoint/ckpt.t7", use_cuda=torch.cuda.is_available()
    )

    # the number of current frame
    num_of_frame = 1

    # a dictionary which stores the history positions of each track
    history_track_positions = {}

    # dics/lists to store the unconfirmed left/right overtaking, confirmed overtaking and their periods
    unconfirmed_left_overtaking = {}
    unconfirmed_right_overtaking = {}
    confirmed_overtaking = []
    confirmed_overtaking_period = []

    with open("results.txt", "w") as f:

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
                not match_across_boundary,
            )

            # convert the bboxes from [x,y,x,y] to [xc,yc,w,h]
            bboxes_all_xcycwh = OD.xyxy2xcycwh(bboxes_all)

            # update deepsort and get the tracking results
            track_outputs = deepsort.update(
                np.array(bboxes_all_xcycwh),
                np.array(classes_all),
                np.array(scores_all),
                im,
                prevent_different_classes_match,
                match_across_boundary,
            )

            # two lists to store the objects that are moving forwards and backwards
            objects_moving_forwards = []
            objects_moving_backwards = []

            # if there are tracked objects in the current frame
            if len(track_outputs) > 0:
                bbox_xyxy = track_outputs[:, :4]
                track_classes = track_outputs[:, 4]
                track_scores = track_outputs[:, 5]
                identities = track_outputs[:, -1]

                # for each track
                for bb_xyxy, track_class, identity in zip(
                    bbox_xyxy, track_classes, identities
                ):

                    # save the tracking results to the txt file
                    f.write(
                        str(num_of_frame)
                        + ","
                        + str(int(identity))
                        + ","
                        + str(deepsort._xyxy_to_tlwh(bb_xyxy))
                        .strip("(")
                        .strip(")")
                        .replace(" ", "")
                        + ","
                        + "-1,-1,-1,-1\n"
                    )

                    # check whether the track is moving forwards or backwards
                    bb_xyxy_list = bb_xyxy.tolist()

                    # if the track is doing an unconfirmed overtake from the left of the image
                    if int(identity) in unconfirmed_left_overtaking:
                        # if the rear of the track has passed the 90 degree, update the overtake to confirmed
                        if bb_xyxy_list[0] >= video_width / 360 * 90:
                            confirmed_overtaking.append(int(identity))
                            confirmed_overtaking_period.append(
                                [
                                    unconfirmed_left_overtaking[int(identity)],
                                    num_of_frame,
                                ]
                            )
                            unconfirmed_left_overtaking.pop(int(identity))
                        # if the front of the track has come back, delete the unconfirmed overtake
                        elif bb_xyxy_list[2] < video_width / 360 * 90:
                            unconfirmed_left_overtaking.pop(int(identity))

                    # if the track is doing an unconfirmed overtake from the right of the image
                    elif int(identity) in unconfirmed_right_overtaking:
                        # if the rear of the track has passed the 270 degree, update the overtake to confirmed
                        if bb_xyxy_list[2] <= video_width / 360 * 270:
                            confirmed_overtaking.append(int(identity))
                            confirmed_overtaking_period.append(
                                [
                                    unconfirmed_right_overtaking[int(identity)],
                                    num_of_frame,
                                ]
                            )
                            unconfirmed_right_overtaking.pop(int(identity))
                        # if the front of the track has come back, delete the unconfirmed overtake
                        elif bb_xyxy_list[0] > video_width / 360 * 270:
                            unconfirmed_right_overtaking.pop(int(identity))

                    # if the track is not doing an overtake and its class is on which we need to detect overtakes
                    if track_class in classes_to_detect_movement:
                        # add the current position of the track to a dictionary called history_track_positions
                        if int(identity) not in history_track_positions.keys():
                            history_track_positions[int(identity)] = [bb_xyxy_list]
                        else:
                            history_track_positions[int(identity)] += [bb_xyxy_list]
                            # count how many times a track moves forwards and backwards in the last five frames
                            if len(history_track_positions[int(identity)]) >= 6:
                                forwards_num = 0
                                backwards_num = 0
                                for ii in range(-6, -1):
                                    if abs(
                                        video_width / 2
                                        - OD.xyxy2xcycwh(
                                            history_track_positions[int(identity)]
                                        )[ii][0]
                                    ) > abs(
                                        video_width / 2
                                        - OD.xyxy2xcycwh(
                                            history_track_positions[int(identity)]
                                        )[ii + 1][0]
                                    ):
                                        forwards_num += 1
                                    else:
                                        backwards_num += 1
                                # if in the last 5 frames, at least 3 frames moves towards the middle line of the image
                                if forwards_num >= 3:
                                    # treat the object as it is moving forwards
                                    objects_moving_forwards.append(int(identity))
                                    # if in the last frame, the front of the track had not passed the 90/270 degree line, but now it has
                                    # give the track an unconfirmed overtaking behaviour
                                    if (
                                        bb_xyxy_list[2] >= video_width / 360 * 90
                                        and history_track_positions[int(identity)][-2][
                                            2
                                        ]
                                        < video_width / 360 * 90
                                    ):
                                        unconfirmed_left_overtaking[
                                            int(identity)
                                        ] = num_of_frame
                                    elif (
                                        bb_xyxy_list[0] <= video_width / 360 * 270
                                        and history_track_positions[int(identity)][-2][
                                            0
                                        ]
                                        > video_width / 360 * 270
                                    ):
                                        unconfirmed_right_overtaking[
                                            int(identity)
                                        ] = num_of_frame
                                # if in the last 5 frames, at least 3 frames moves away from the middle line of the image
                                elif backwards_num >= 3:
                                    # treat the object as it is moving backwards
                                    objects_moving_backwards.append(int(identity))

                # if the function is used for unconfirmed overtaking behaviour detection, draw the tracks with the overtaking boxes
                if mode == "Unconfirmed":
                    im = draw_boxes(
                        im,
                        bbox_xyxy,
                        track_classes,
                        track_scores,
                        video_width,
                        identities,
                        objects_moving_backwards,
                        objects_moving_forwards,
                        unconfirmed_left_overtaking,
                        unconfirmed_right_overtaking,
                        size_thresholds,
                        True,
                        classes_to_detect_movement,
                    )

                # if the function is used for confirmed overtaking behaviour detection, only draw the tracks
                elif mode == "Confirmed":
                    im = draw_boxes(
                        im,
                        bbox_xyxy,
                        track_classes,
                        track_scores,
                        video_width,
                        identities,
                        objects_moving_backwards,
                        objects_moving_forwards,
                    )

            # save the frame to the output file
            outputfile.write(im)

            # show the current FPS
            time2 = time.time()
            if num_of_frame % 5 == 0:
                print(num_of_frame, "/", video_frame_count)
                print(str(1 / (time2 - time1)) + " fps")

            num_of_frame += 1

    # release the input and output videos
    video_capture.release()
    outputfile.release()

    # since the confirmed overtakes can only be detected after the whole behaviour has been finished
    # in the 'Confirmed' mode, draw the boxes for comfirmed overtakes after the process of detection
    if mode == "Confirmed":

        print("Confirmed overtaking tracks:", confirmed_overtaking)
        print("Confirmed overtaking periods:", confirmed_overtaking_period)

        # copy and paste the output video with tracking results
        v_src = open(output_video_path, "rb")
        content = v_src.read()
        v_copy = open("copy.mp4", "wb")
        v_copy.write(content)
        v_src.close()
        v_copy.close()

        video_capture = cv2.VideoCapture("copy.mp4")
        video_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        outputfile = cv2.VideoWriter(
            output_video_path, fourcc, video_fps, (video_width, video_height)
        )
        num_of_frame = 1

        tracking_results = []

        # read the tracking results
        with open("results.txt", "r") as f:
            data = f.readlines()
            for line in data:
                tracking_results.append(line)

        # for each image frame in the video
        while video_capture.grab():
            # get the next image frame
            _, im = video_capture.retrieve()

            # when an track is between the start and end frames of an confirmed overtaking behaviour
            for i, q in zip(confirmed_overtaking_period, confirmed_overtaking):
                # color the bbox of the track with red
                if num_of_frame in range(i[0], i[1]):
                    for line in tracking_results:
                        contents = line.split(",")
                        if contents[0] == str(num_of_frame) and contents[1] == str(q):
                            red_area = np.zeros(im.shape, np.uint8)
                            cv2.rectangle(
                                red_area,
                                (int(float(contents[2])), int(float(contents[3]))),
                                (
                                    int(float(contents[2])) + int(float(contents[4])),
                                    int(float(contents[3])) + int(float(contents[5])),
                                ),
                                (0, 0, 255),
                                -1,
                            )
                            im = cv2.addWeighted(im, 1.0, red_area, 0.5, 1)
            outputfile.write(im)
            if num_of_frame % 5 == 0:
                print(num_of_frame, "/", video_frame_count)
            num_of_frame += 1

        # release the videos again
        video_capture.release()
        outputfile.release()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video_path", required=True, type=str)
    parser.add_argument("--output_video_path", required=True, type=str)
    parser.add_argument(
        "--mode", type=str, choices=["Confirmed", "Unconfirmed"], default="Confirmed"
    )
    parser.add_argument(
        "--prevent_different_classes_match", default=True, type=boolean_string
    )
    parser.add_argument("--match_across_boundary", default=True, type=boolean_string)
    parser.add_argument(
        "--classes_to_detect", nargs="+", type=int, default=[0, 1, 2, 3, 5, 7, 9]
    )
    parser.add_argument(
        "--classes_to_detect_movement", nargs="+", type=int, default=[2, 5, 7]
    )

    parser.add_argument(
        "--size_thresholds",
        nargs="+",
        type=int,
        default=[500 * 500, 900 * 900, 600 * 600],
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
    print(opt)
    return opt


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def main(opt):
    Overtaking_Detection(
        opt.input_video_path,
        opt.output_video_path,
        opt.mode,
        opt.prevent_different_classes_match,
        opt.match_across_boundary,
        opt.classes_to_detect,
        opt.classes_to_detect_movement,
        opt.size_thresholds,
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
