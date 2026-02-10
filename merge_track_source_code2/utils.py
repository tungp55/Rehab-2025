from itertools import chain
import numpy as np
import cv2
try:
    import simplejson
except ImportError:
    import json as simplejson
import json
from networkx.readwrite import json_graph
from tqdm import tqdm

class timecode():
    def __init__(self, start_time, stop_time):
        self.start_time = start_time
        self.stop_time = stop_time


def ranges_intersection(range1, range2):
    start = max(range1[0], range2[0])
    stop = min(range1[1], range2[1])
    if start >= stop:
        return 0, 0
    return start, stop


def ranges_is_overlap(range1, range2):
    if (range1[0] < range2[0] and range1[1] > range2[0]):
        return True
    if (range1[1] < range2[1] and range1[1] > range2[0]):
        return True
    if (range2[0] < range1[1] and range2[1] > range1[0]):
        return True
    if (range2[1] < range1[1] and range2[1] > range1[0]):
        return True
    return False


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


class box:
    def __init__(self,
                 x1,
                 y1,
                 width,
                 height
                 ):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x1 + width
        self.y2 = y1 + height


class frame_data:
    def __init__(self,
                 frame_id,
                 track_id,
                 box,
                 score=1.0,
                 class_id=1,
                 visibility=1
                 ):
        self.frame_id = frame_id
        self.track_id = track_id
        self.bounding_box = box
        self.score = score
        self.class_id = class_id
        self.visibility = visibility


class track_data:
    def __init__(self,
                 start_frame,
                 stop_frame,
                 box
                 ):
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.last_bounding_box = box


def centroid_distance(tracks_data, u, v):
    u_centroid_x = (tracks_data[u].last_bounding_box.x1 +
                    tracks_data[u].last_bounding_box.x2 / 2)
    u_centroid_y = (tracks_data[u].last_bounding_box.y1 +
                    tracks_data[u].last_bounding_box.y2 / 2)
    v_centroid_x = (tracks_data[v].last_bounding_box.x1 +
                    tracks_data[v].last_bounding_box.x2 / 2)
    v_centroid_y = (tracks_data[v].last_bounding_box.y1 +
                    tracks_data[v].last_bounding_box.y2 / 2)
    point1 = np.array((1, u_centroid_x, u_centroid_y))
    point2 = np.array((1, v_centroid_x, v_centroid_y))
    dist = np.linalg.norm(point1 - point2)
    return dist


def write_tracks_chain_to_file(frames_data, chain_tracks_hands, fname,
                               start_tracking_time_seconds, stop_tracking_time_seconds, frame_rate, tracks_data):
    nframes = (stop_tracking_time_seconds -
               start_tracking_time_seconds) * frame_rate
    nframes = 25000
    offset_nframes = start_tracking_time_seconds * frame_rate
    track_dic = {}
    for hand_side, chain_tracks in chain_tracks_hands:
        for track_id in chain_tracks:
            track_dic[track_id] = hand_side
    
    # Find the next available track_id for non-merged tracks
    if track_dic:
        next_track_id = max(track_dic.values()) + 1
    else:
        next_track_id = 1
    
    writer = open(fname, "w+")
    for frame_id in tqdm(range(1, nframes + 2)):
        if frame_id == 361:
            xx=0
        frames = [frame for frame in frames_data
                  if frame.frame_id == frame_id + offset_nframes]
        for frame in frames:
            if frame.track_id in track_dic.keys():  # nếu có thì lấy từ dictionary
                track_id = track_dic[frame.track_id]
            else:  # nếu chưa thì thêm vào dictionary
                track_dic[frame.track_id] = next_track_id
                track_id = next_track_id
                next_track_id += 1
            bounding_box = frame.bounding_box
            # Ghi đầy đủ 9 cột: frame_id, track_id, x1, y1, width, height, score, class_id, visibility
            score = getattr(frame, 'score', 1.0)
            class_id = getattr(frame, 'class_id', 1)
            visibility = getattr(frame, 'visibility', 1)
            line_string = "{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(
                frame_id, track_id, 
                bounding_box.x1, bounding_box.y1,
                bounding_box.x2 - bounding_box.x1, bounding_box.y2 - bounding_box.y1,
                score, class_id, visibility
            )
            writer.write(line_string)

        # for hand_side, chain_tracks in chain_tracks_hands:
        #     tracks_by_frame = [track_id for track_id in chain_tracks
        #                        if tracks_data[track_id].start_frame <= frame_id + offset_nframes
        #                        and tracks_data[track_id].stop_frame >= frame_id + offset_nframes]
        #     for track_id in tracks_by_frame:
        #         frames = [frame for frame in frames_data
        #                   if frame.frame_id == frame_id + offset_nframes and frame.track_id == track_id]
        #         for frame in frames:
        #             bounding_box = frame.bounding_box
        #             line_string = "{}, {}, {}, {}, {}, {}\n".format(frame_id, hand_side, bounding_box.x1, bounding_box.y1,
        #                                                             bounding_box.x2 - bounding_box.x1, bounding_box.y2 - bounding_box.y1)
        #             writer.write(line_string)
    writer.close()


def write_deep_sort_data(frames_data, fname,
                         start_tracking_time_seconds, stop_tracking_time_seconds, frame_rate, tracks_data,
                         fix_tracks):
    nframes = (stop_tracking_time_seconds -
               start_tracking_time_seconds) * frame_rate
    offset_nframes = start_tracking_time_seconds * frame_rate
    writer = open(fname, "w+")
    track_dic = {}  # mapping deep_sort_track_id -> merge_track_track_id
    for hand_side, fix_track in fix_tracks:
        track_dic[fix_track] = hand_side
    for frame_id in range(1, nframes + 2):
        frames = [frame for frame in frames_data
                      if frame.frame_id == frame_id + offset_nframes]
        for frame in frames:
            if frame.track_id in track_dic.keys():  # nếu có thì lấy từ dictionary
                    track_id = track_dic[frame.track_id]
            else:  # nếu chưa thì thêm vào dictionary
                new_track_id = max(track_dic.values()) + 1
                track_dic[frame.track_id] = new_track_id
                track_id = new_track_id
            bounding_box = frame.bounding_box
            line_string = "{}, {}, {}, {}, {}, {}\n".format(frame_id, track_id, bounding_box.x1, bounding_box.y1,
                                                                bounding_box.x2 - bounding_box.x1, bounding_box.y2 - bounding_box.y1)
            writer.write(line_string)
    writer.close()


def save_graph_2_json(G, fname):
    simplejson.dump(dict(nodes=[[n, G.nodes[n]] for n in G.nodes],
                         edges=[[u, v, G.edges[u][v]] for u, v in G.edges()]),
                    open(fname, 'w'), indent=2)


def save_graph(G, fname):
    with open(fname, 'w') as outfile1:
        outfile1.write(json.dumps(G))
