from typing import Sized
import cv2
import numpy as np
from datetime import date, datetime, time
import numpy


class timecode():
    def __init__(self, start_time, stop_time):
        self.start_time = start_time
        self.stop_time = stop_time


class point():
    def __init__(self, x, y):
        self.x = x
        self.y = y


VIDEO_FILES = ['./data3/GH010373_data3_small.MP4']
PRED_FILE = './data3/pred_v3.txt'
GT_FILE = './data3/gt.txt'
# OUTPUT_FILE = './data2/data2_v3.mp4'
OUTPUT_FILE = ''
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


def readTimeCodes(fName):
    left_hand = []
    right_hand = []
    file1 = open(fName, 'r')
    Lines = file1.readlines()
    count = 0
    for line in Lines:
        count += 1
        if count > 1:
            vals = line.strip().split()
            if vals[2] == '1':
                left_hand.append(timecode(float(vals[0]), float(vals[1])))
            else:
                right_hand.append(timecode(float(vals[0]), float(vals[1])))
    return left_hand, right_hand


def plotPoints(timecodes, bottom_line, line_height):
    pts = [[0, bottom_line]]
    for code in timecodes:
        start_x = int(code.start_time * frame_width /
                      (nr_of_frames / frame_rate))
        stop_x = int(code.stop_time * frame_width /
                     (nr_of_frames / frame_rate))
        pts.append([start_x, bottom_line])
        pts.append([start_x, bottom_line - line_height])
        pts.append([stop_x, bottom_line - line_height])
        pts.append([stop_x, bottom_line])
    pts.append([frame_width, bottom_line])
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    return pts


def check_activate_hand(elapsed_time, left_hand, right_hand):
    left_activate = right_activate = False
    for time_code in left_hand:
        if time_code.start_time <= elapsed_time and time_code.stop_time >= elapsed_time:
            left_activate = True
    for time_code in right_hand:
        if time_code.start_time <= elapsed_time and time_code.stop_time >= elapsed_time:
            right_activate = True
    return left_activate, right_activate


def get_pointer_x(elapsed_time):
    pointer_x = int(elapsed_time * frame_width / (nr_of_frames / frame_rate))
    return pointer_x


def rectangle_pointers(x, y):
    pts = [[x, y],
           [x - 7, y + 7],
           [x + 7, y + 7]]
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    return pts


start_play_time = datetime.now()
# open video
videos = []
for video_file_name in VIDEO_FILES:
    vid = cv2.VideoCapture(video_file_name)
    videos.append(vid)
# get frame rate
frame_rate = int(videos[0].get(cv2.CAP_PROP_FPS))
# get total number of frames
nr_of_frames = 0
for video in videos:
    nr_of_frames += int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(videos[0].get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(videos[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
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

pred_left_hand_pts = plotPoints(pred_left_hand, LEFT_BOTTOM_Y, LINE_HEIGHT)
pred_right_hand_pts = plotPoints(pred_right_hand, RIGHT_BOTTOM_Y, LINE_HEIGHT)

gt_left_hand_pts = plotPoints(gt_left_hand, LEFT_BOTTOM_Y, LINE_HEIGHT + 5)
gt_right_hand_pts = plotPoints(gt_right_hand, RIGHT_BOTTOM_Y, LINE_HEIGHT + 5)

# left_hand_pts = np.array(left_hand_pts, np.int32)
# left_hand_pts = left_hand_pts.reshape((-1, 1, 2))

# right_hand_pts = np.array(right_hand_pts, np.int32)
# right_hand_pts = right_hand_pts.reshape((-1, 1, 2))
# total milisecond unsync
hist_miss = 0
# total milisecend on 1 frame
milisecond_per_frame = 1000 / frame_rate
# output video
# out = cv2.VideoWriter('capture.avi', cv2.VideoWriter_fourcc(
#     'M', 'J', 'P', 'G'), frame_rate, (frame_width, frame_height))
out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(
    *'MP4V'), frame_rate, (frame_width, frame_height))

pointer_on_left = True
video_counter = 0
video = videos[video_counter]
hist_video_frames = 0
# main loop
while 1 and video_counter < len(videos):
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
    pointer_x = get_pointer_x(elapsed_time_seconds)
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
        pts = rectangle_pointers(pointer_x, LEFT_BOTTOM_Y)
        cv2.fillPoly(frame, [pts], POINTER_COLOR)
    if is_right or (not is_left and not is_right and not pointer_on_left):
        pts = rectangle_pointers(pointer_x, RIGHT_BOTTOM_Y)
        cv2.fillPoly(frame, [pts], POINTER_COLOR)
    # show frame, break the loop if no frame is found
    if ret:
        cv2.imshow("Video", frame)
        out.write(frame)
        # update slider position on trackbar
        # NOTE: this is an expensive operation, remove to greatly increase max playback speed
        cv2.setTrackbarPos("Frame", "Video", int(
            video.get(cv2.CAP_PROP_POS_FRAMES)))
    elif video_counter < len(videos) - 1:
            video_counter += 1
            video = videos[video_counter]
            hist_video_frames += video.get(cv2.CAP_PROP_FRAME_COUNT)
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
