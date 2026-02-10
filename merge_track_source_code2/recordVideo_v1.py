from typing import Sized
import cv2
import numpy as np
from datetime import date, datetime, time
import numpy
from utils import timecode, readTimeCodes
from video_utils import draw_text, triangle_pointers, plotPoints, get_pointer_x
from merge_tracks import merge_tracks

class point():
    def __init__(self, x, y):
        self.x = x
        self.y = y


# VIDEO_FILE = './data3/GH010373_data3_small.MP4'
VIDEO_FILE = 'J:/MITICA/GH010373_data3_small.MP4'
PRED_FILE = './data3/pred.txt'
GT_FILE = './data3/gt.txt'
OUTPUT_FILE = 'J:/MITICA/data3/acc_deep_sort pred1.MP4'
# OUTPUT_FILE = ''

LINE_HEIGHT = 10
LEFT_BOTTOM_Y = 50
RIGHT_BOTTOM_Y = 80
LEFT_COLOR = (255, 0, 0)
RIGHT_COLOR = (0, 255, 0)
GT_COLOR = (255, 255, 255)
POINTER_COLOR = (0, 0, 255)


def getFrame(frame_nr):
    global video
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)

#  function called by trackbar, sets the speed of playback


def setSpeed(val):
    global playSpeed
    playSpeed = max(val, 1)


def check_activate_hand(elapsed_time, left_hand, right_hand):
    left_activate = right_activate = False
    for time_code in left_hand:
        if time_code.start_time <= elapsed_time and time_code.stop_time >= elapsed_time:
            left_activate = True
    for time_code in right_hand:
        if time_code.start_time <= elapsed_time and time_code.stop_time >= elapsed_time:
            right_activate = True
    return left_activate, right_activate


def draw_deep_sort(resolution_rate, frames_data, shift_nframes=0,
                   img_frame=None, video=None):
    BOX_COLOR = (255, 128, 0)
    current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
    elapsed_time_seconds = current_frame / frame_rate
    # Get the next videoframe
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = int(4 * resolution_rate)
    font_text_color = (0, 255, 255)
    font_bgr_color = (0, 204, 102)
    font_thickness = int(4 * resolution_rate)
    line_type = 1
    # draw tracking bounding box
    track_data_org = [
        frame for frame in frames_data if frame.frame_id == current_frame + shift_nframes + 1]
    for tracking_gt_frame in track_data_org:
        tracking_id = tracking_gt_frame.track_id
        bounding_box = tracking_gt_frame.bounding_box
        pt1 = (int(bounding_box.x1 * resolution_rate),
               int(bounding_box.y1 * resolution_rate))
        pt2 = (int(bounding_box.x2 * resolution_rate),
               int(bounding_box.y2 * resolution_rate))
        cv2.rectangle(img_frame, pt1, pt2,
                      BOX_COLOR, thickness=5)
        draw_text(img_frame, str(tracking_id),
                  'auto',
                  font, bounding_box, resolution_rate,
                  font_scale, font_thickness, font_text_color, font_bgr_color)

if __name__ == '__main__':
    merger = merge_tracks()
    merger.TRACK_DATA_FILE = 'F:/MITICA/merge tracks/Thay Hai/out_txts/GH010373.txt'
    # merger.TRACK_DATA_FILE = 'J:/MITICA/merge tracks/GH010373_260_380_vung_nhanh/deep_sort.txt'
    merger.START_TRACKING_TIME_SECONDS = 0
    merger.STOP_TRACKING_TIME_SECONDS = 120
    merger.init()

    start_play_time = datetime.now()
    # open video
    video = cv2.VideoCapture(VIDEO_FILE)
    # get frame rate
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    # get total number of frames
    nr_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # create display window
    cv2.namedWindow("Video")
    _, __, window_width, window_height = cv2.getWindowImageRect('Video')

    # cv2.resizeWindow('Video', 1000, 1000)
    # set wait for each frame, determines playbackspeed
    playSpeed = 100
    cv2.createTrackbar("Speed", "Video", playSpeed, 200, setSpeed)
    # add trackbar
    window = cv2.createTrackbar("Frame", "Video", 0, nr_of_frames, getFrame)
    pred_left_hand, pred_right_hand = readTimeCodes(PRED_FILE)
    gt_left_hand, gt_right_hand = readTimeCodes(GT_FILE)


    pred_left_hand_pts = plotPoints(pred_left_hand, LEFT_BOTTOM_Y, LINE_HEIGHT,
                                    frame_width, frame_rate, nr_of_frames)
    pred_right_hand_pts = plotPoints(pred_right_hand, RIGHT_BOTTOM_Y, LINE_HEIGHT,
                                    frame_width, frame_rate, nr_of_frames)

    gt_left_hand_pts = plotPoints(gt_left_hand, LEFT_BOTTOM_Y, LINE_HEIGHT + 5,
                                frame_width, frame_rate, nr_of_frames)
    gt_right_hand_pts = plotPoints(gt_right_hand, RIGHT_BOTTOM_Y, LINE_HEIGHT + 5,
                                frame_width, frame_rate, nr_of_frames)

    # total milisecend on 1 frame
    milisecond_per_frame = 1000 / frame_rate
    # output video
    # out = cv2.VideoWriter('capture.avi', cv2.VideoWriter_fourcc(
    #     'M', 'J', 'P', 'G'), frame_rate, (frame_width, frame_height))
    out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(
        *'MP4V'), frame_rate, (frame_width, frame_height))

    pointer_on_left = True
    hist_video_frames = 0
    # main loop
    while 1:
        current_frame = hist_video_frames + video.get(cv2.CAP_PROP_POS_FRAMES)
        elapsed_time_seconds = current_frame / frame_rate
        # Get the next videoframe
        ret, frame = video.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (20, 20)
        fontScale = 0.5
        fontColor = (255, 255, 255)
        lineType = 1
        text_time = '{:02}:{:02}:{:.2f}'.format(int(elapsed_time_seconds / 60), int(
            elapsed_time_seconds) % 60, elapsed_time_seconds - int(elapsed_time_seconds))
        cv2.polylines(frame, [gt_left_hand_pts], isClosed=False, color=GT_COLOR)
        cv2.polylines(frame, [gt_right_hand_pts], isClosed=False, color=GT_COLOR)

        cv2.polylines(frame, [pred_left_hand_pts],
                    isClosed=False, color=LEFT_COLOR)
        cv2.polylines(frame, [pred_right_hand_pts],
                    isClosed=False, color=RIGHT_COLOR)

        cv2.putText(frame, text_time,
                    (bottomLeftCornerOfText),
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        cv2.putText(frame, 'Left hand',
                    (150, 20),
                    font,
                    fontScale,
                    LEFT_COLOR,
                    lineType)
        cv2.putText(frame, 'Right hand',
                    (250, 20),
                    font,
                    fontScale,
                    RIGHT_COLOR,
                    lineType)
        cv2.putText(frame, 'Ground truth',
                    (350, 20),
                    font,
                    fontScale,
                    GT_COLOR,
                    lineType)
        is_left, is_right = check_activate_hand(
            elapsed_time_seconds, pred_left_hand, pred_right_hand)
        pointer_x = get_pointer_x(elapsed_time_seconds,
                                frame_width, nr_of_frames, frame_rate)
        if is_left:
            cv2.putText(frame, 'Left hand in action',
                        (50, 100),
                        font,
                        fontScale,
                        LEFT_COLOR,
                        2)
            pointer_on_left = True
            # cv2.circle(frame, (pointer_x, LEFT_BOTTOM_Y), 3, (0, 0, 255), 3)
        if is_right:
            cv2.putText(frame, 'Right hand in action',
                        (300, 100),
                        font,
                        fontScale,
                        RIGHT_COLOR,
                        2)
            pointer_on_left = False
            # cv2.circle(frame, (pointer_x, RIGHT_BOTTOM_Y), 3, (0, 0, 255), 3)
        if is_left or (not is_left and not is_right and pointer_on_left):
            pts = triangle_pointers(pointer_x, LEFT_BOTTOM_Y)
            cv2.fillPoly(frame, [pts], POINTER_COLOR)
        if is_right or (not is_left and not is_right and not pointer_on_left):
            pts = triangle_pointers(pointer_x, RIGHT_BOTTOM_Y)
            cv2.fillPoly(frame, [pts], POINTER_COLOR)

        draw_deep_sort(resolution_rate=0.25, frames_data=merger.frames_data, img_frame=frame,video=video)
        # show frame, break the loop if no frame is found
        if ret:
            cv2.imshow("Video", frame)
            out.write(frame)
            # update slider position on trackbar
            # NOTE: this is an expensive operation, remove to greatly increase max playback speed
            cv2.setTrackbarPos("Frame", "Video", int(
                video.get(cv2.CAP_PROP_POS_FRAMES)))
        else:
            break
        key = cv2.waitKey(10)
        # stop playback when q is pressed
        if key == ord('q'):
            break
    # release resources
    video.release()
    out.release()
    cv2.destroyAllWindows()
