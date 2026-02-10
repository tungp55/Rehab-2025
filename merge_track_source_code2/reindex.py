from io import StringIO
import cv2
import numpy as np
from datetime import date, datetime, time
import numpy

DATA_FILE = './GH010371.txt'


class track_data:
    def __init__(self, track_id,
                 x1,
                 y1,
                 x2,
                 y2,
                 start_frame,
                 stop_frame
                 ):
        self.track_id = track_id
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.start_frame = start_frame
        self.stop_frame = stop_frame

data_lines = []


def read_frame_data(fName):
    left_hand = []
    right_hand = []
    file1 = open(fName, 'r')
    Lines = file1.readlines()
    count = 0
    for line in Lines:
        count += 1
        if count > 1:
            vals = line.strip().split()
            data_lines.append(vals)
    return data_lines

def convert_2_track_data(lines):
    tracks = []
    max_track_id = 0
    for vals in lines:
        frame_id = vals[0]
        track_id = vals[1]
        x1 = vals[2]
        y1 = vals[3]
        x2 = vals[4]
        y2 = vals[5]
        max_track_id = max(max_track_id, track)
        t = [track for track in tracks if track.track_id == track_id]
        #if exists
        if t:
            t.stop_frame = frame_id
        else:    
            tracks.append(track_data(track_id, x1, y1, x2, y2, frame_id, frame_id))
        

