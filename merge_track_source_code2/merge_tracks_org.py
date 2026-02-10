from io import StringIO
import cv2
from networkx.algorithms.cycles import simple_cycles
import numpy as np
from datetime import date, datetime, time
from itertools import chain, combinations
import networkx as nx
# import bellmanford as bf  # Replaced with NetworkX built-in
from utils import *
from video_utils import visualize_chain_tracks
from networkx.readwrite import json_graph
from merge_tracks_convert_labelme import *


class merge_tracks:
    FRAME_WIDTH = 1920
    FRAME_RATE = 30
    VIDEO_FILE = '/home/mcn/bachnh/merge_tracks/videos/GH010373_380_500_che_khuat_resol05.MP4'
    ACC_DATA_FILE = './data2/pred v2.txt'
    # TRACK_DATA_FILE = '../tracking reindex/GH010371.txt'
    TRACK_DATA_FILE = '/home/mcn/bachnh/merge_tracks/merge tracks/Thay Hai/out_txts/GH010373.txt'
    FIX_TRACK_ID = 140
    FIX_HAND_SIDE = 1  # 1 trái, 2 phải
    THRESHOLE_SWITCH_CENTROIDS = 0
    DELTA_T_SECONDS = 10
    # delta_t: thời gian để tính tỉ lệ activate chuyển tiếp giữa 2 track
    START_TRACKING_TIME_SECONDS = 170
    STOP_TRACKING_TIME_SECONDS = 290
    # khoảng thời gian tính reindex trên dữ liệu tracking

    tracks_data = []
    frames_data = []
    accept_sols = []
    max_track_id = 0
    max_centroid_distance = 0
    max_track_lifetime = 0

    def read_frame_data(self, fName):
        data_lines = []
        file1 = open(fName, 'r')
        Lines = file1.readlines()
        for line in Lines:
            vals = line.strip().split(',')
            data_lines.append(vals)
        return data_lines

    def frameId_in_tracking_time(self, frame_id):
        return (frame_id >= self.START_TRACKING_TIME_SECONDS * self.FRAME_RATE
                and (frame_id <= self.STOP_TRACKING_TIME_SECONDS * self.FRAME_RATE
                     or self.STOP_TRACKING_TIME_SECONDS == 0))

    def convert_2_track_data(self, lines):
        tracks = []
        max_track_id = 0
        for vals in lines:
            frame_id = int(float(vals[0]))
            if frame_id == 361:
                xx=0
            track_id = int(float(vals[1]))
            x1 = int(float(vals[2]))
            y1 = int(float(vals[3]))
            width = int(float(vals[4]))
            height = int(float(vals[5]))
            # Đọc các giá trị bổ sung nếu có
            score = float(vals[6]) if len(vals) > 6 else 1.0
            class_id = int(float(vals[7])) if len(vals) > 7 else 1
            visibility = int(float(vals[8])) if len(vals) > 8 else 1
            bounding_box = box(x1, y1, width, height)
            if (not self.START_TRACKING_TIME_SECONDS or not self.STOP_TRACKING_TIME_SECONDS or
                    (self.frameId_in_tracking_time(frame_id))):
                frame = frame_data(frame_id, track_id, bounding_box, score, class_id, visibility)
                self.frames_data.append(frame)
                max_track_id = max(max_track_id, track_id)
                for i in range(0, max_track_id + 1 - len(tracks)):
                    tracks.append(None)
                # if exists
                if tracks[track_id]:
                    tracks[track_id].stop_frame = frame_id
                else:
                    tracks[track_id] = track_data(
                        frame_id, frame_id, bounding_box)
                tracks[track_id].last_box = bounding_box
        return tracks, max_track_id, self.frames_data

    def box_side(self, box):
        right_intersection = ranges_intersection((box.x1, box.x2),
                                                 ((int(self.FRAME_WIDTH / 2), self.FRAME_WIDTH)))
        left_intersection = ranges_intersection((box.x1, box.x2),
                                                (0, int(self.FRAME_WIDTH / 2)))
        if right_intersection[1] - right_intersection[0] > left_intersection[1] - left_intersection[0]:
            return 2
        else:
            return 1

    def tracks_is_overlap(self, track_id_1, track_id_2):
        if self.box_side(self.tracks_data[track_id_2].last_bounding_box) != self.FIX_HAND_SIDE:
            return True
        if centroid_distance(self.tracks_data, track_id_1, track_id_2) > self.THRESHOLE_SWITCH_CENTROIDS:
            return True
        if ((track_id_1 >= len(self.tracks_data) or track_id_2 >= len(self.tracks_data)) or
                (not self.tracks_data[track_id_2]) or (not self.tracks_data[track_id_1])):
            return True
        return ranges_is_overlap((self.tracks_data[track_id_1].start_frame,
                                  self.tracks_data[track_id_1].stop_frame),
                                 (self.tracks_data[track_id_2].start_frame,
                                 self.tracks_data[track_id_2].stop_frame))

    def get_chain_tracks(self, sol):
        last_track_id = sol[len(sol) - 1]
        for i in range(last_track_id + 1, self.max_track_id + 1):
            if self.tracks_data[i]:
                if (not self.tracks_is_overlap(last_track_id, i)):
                    new_sol = sol.copy()
                    new_sol.append(i)
                    self.accept_sols.append(new_sol)
                    print(new_sol)
                    self.get_chain_tracks(new_sol)
        # sol.pop()
        if len(sol) < 1:
            return

    def calculate_maximum_values(self):
        max_track_lifetime = 0
        for i in range(self.FIX_TRACK_ID + 1, self.max_track_id):
            if (self.tracks_data[i]):
                max_track_lifetime = max(
                    max_track_lifetime, self.tracks_data[i].stop_frame - self.tracks_data[i].start_frame)
        max_centroid_distance = 0
        for u in range(self.FIX_TRACK_ID, self.max_track_id - 1):
            for v in range(u + 1, self.max_track_id):
                if (self.tracks_data[u] and self.tracks_data[v]):
                    max_centroid_distance = max(
                        max_centroid_distance, centroid_distance(self.tracks_data, u, v))
        return max_track_lifetime, max_centroid_distance

    def centroid_score(self, u, v):
        return 1 - (centroid_distance(self.tracks_data, u, v) / self.max_centroid_distance)

    def lifetime_score(self, u):
        if self.max_track_lifetime == 0:
            return 0
        return (self.tracks_data[u].stop_frame - self.tracks_data[u].start_frame) / self.max_track_lifetime

    def switch_range(self, range1, range2):
        start = max(range1[0], range2[0])
        stop = min(range1[1], range2[1])
        if start >= stop:
            return 0
        return start, stop

    def switch_tracks_on_activate(self, u, v, acc_action_hand_data, frame_rate):
        if self.tracks_is_overlap(u, v):
            return 0
        start_intersection_frame, stop_intersection_frame = ranges_intersection((self.tracks_data[u].start_frame,
                                                                                self.tracks_data[u].stop_frame),
                                                                                (self.tracks_data[v].start_frame,
                                                                                self.tracks_data[v].stop_frame))
        return (stop_intersection_frame - start_intersection_frame) / frame_rate / self.DELTA_T_SECONDS

    def cost(self, u, v):
        if (self.tracks_is_overlap(u, v)):
            return 1000
        return - self.lifetime_score(v) - self.switch_tracks_on_activate(
            u, v, 
            # self.acc_left_hand if self.FIX_HAND_SIDE == 1 else self.acc_right_hand,
            None,
            self.FRAME_RATE)
        # centroid_score(u, v)
        # TODO add ACC score

    def build_graph(self):
        G = nx.DiGraph()
        for i in range(1, self.max_track_id + 1):
            G.add_node(i)
        for u in range(1, self.max_track_id - 1):
            for v in range(u + 1, self.max_track_id):
                if self.tracks_data[u] and self.tracks_data[v]:
                    if not self.tracks_is_overlap(u, v):
                        edge_cost = self.cost(u, v)
                        G.add_edge(u, v, length=edge_cost)
        for i in range(2, self.max_track_id):
            G.add_edge(i, self.max_track_id + 1, length=0)
        return G

    def get_weight(self, path):
        return len(path)

    def init(self):
        # self.acc_left_hand, self.acc_right_hand = readTimeCodes(
        #     self.ACC_DATA_FILE)
        # Reset class variables to avoid accumulation across instances
        self.frames_data = []
        self.tracks_data = []
        self.accept_sols = []
        self.max_track_id = 0
        self.max_centroid_distance = 0
        self.max_track_lifetime = 0
        self.lines_data = self.read_frame_data(self.TRACK_DATA_FILE)
        self.tracks_data, self.max_track_id, self.frames_data = self.convert_2_track_data(
            self.lines_data)
        self.tracks_data = self.tracks_data[:self.max_track_id]
        self.max_track_lifetime, self.max_centroid_distance = self.calculate_maximum_values()


if __name__ == '__main__':
    # convert_json_2_tracks_file('F:/MITICA/merge tracks/GH010371_170_290_tay_bac_sy/GH010371_170_290_annotations/',
    # 'F:/MITICA/merge tracks/GH010371_170_290_tay_bac_sy/gt.txt')

    merger = merge_tracks()
    merger.ACC_DATA_FILE = '/home/mcn/bachnh/merge_tracks/visualize GT result/data3/pred v2.txt'
    merger.VIDEO_FILE = 'D:/research/mitica/2023/tiny/input/GH010373_data3_small.MP4'
    merger.TRACK_DATA_FILE = 'D:/research/mitica/2023/tiny/input/deepsort_output_GH010373_data3_small.txt'
    merger.START_TRACKING_TIME_SECONDS = 0
    merger.STOP_TRACKING_TIME_SECONDS = 0

    fix_tracks = [(1, 1), (2, 2)]

    merger.FIX_TRACK_ID = fix_tracks[0][1]
    merger.FIX_HAND_SIDE = fix_tracks[0][0]  # 1 trái, 2 phải

    merger.init()
    graph = merger.build_graph()
    # Use NetworkX's built-in Bellman-Ford algorithm
    path_length = nx.bellman_ford_path_length(
        graph, source=merger.FIX_TRACK_ID, target=merger.max_track_id + 1, weight="length")
    path_nodes = nx.bellman_ford_path(
        graph, source=merger.FIX_TRACK_ID, target=merger.max_track_id + 1, weight="length")
    path_nodes.pop()
    print(path_length)
    print(path_nodes)
    left_nodes = path_nodes.copy()

    merger = merge_tracks()
    merger.VIDEO_FILE = 'D:/research/mitica/2023/tiny/input/GH010373_data3_small.MP4'
    merger.TRACK_DATA_FILE = 'D:/research/mitica/2023/tiny/input/deepsort_output_GH010373_data3_small.txt'
    merger.START_TRACKING_TIME_SECONDS = 0
    merger.STOP_TRACKING_TIME_SECONDS = 0
    merger.FIX_TRACK_ID = fix_tracks[1][1]
    merger.FIX_HAND_SIDE = fix_tracks[1][0]  # 1 trái, 2 phải
    merger.init()
    graph = merger.build_graph()
    # Use NetworkX's built-in Bellman-Ford algorithm
    path_length = nx.bellman_ford_path_length(
        graph, source=merger.FIX_TRACK_ID, target=merger.max_track_id + 1, weight="length")
    path_nodes = nx.bellman_ford_path(
        graph, source=merger.FIX_TRACK_ID, target=merger.max_track_id + 1, weight="length")
    path_nodes.pop()
    print(path_length)
    print(path_nodes)
    right_nodes = path_nodes.copy()

    # write_tracks_chain_to_file(merger.frames_data, [(
    #     1, left_nodes), (2, right_nodes)],
    #     'D:/research/mitica/2023/tiny/input/GH010373_data3_small_merge_tracks_2_hands.txt',
    #     merger.START_TRACKING_TIME_SECONDS, merger.STOP_TRACKING_TIME_SECONDS,
    #     merger.FRAME_RATE, merger.tracks_data)

    # write_deep_sort_data(merger.frames_data,
    #                      './GH010373_380_500_che_khuat_deep_sort.txt',
    #                      merger.START_TRACKING_TIME_SECONDS, merger.STOP_TRACKING_TIME_SECONDS,
    #                      merger.FRAME_RATE, merger.tracks_data,
    #                      fix_tracks
    #                      )

    visualize_chain_tracks('merge_tracks_center',
                           video_fname=merger.VIDEO_FILE,
                           #    output_video_fname,
                        #    output_fname='J:/MITICA/merge tracks/GH010373_260_380_vung_nhanh/right_hand.MP4',
                              output_fname='',
                           resolution_rate=1,
                        #    acc_action_hand_data=merger.acc_left_hand if merger.FIX_HAND_SIDE == 1 else merger.acc_right_hand,
                            acc_action_hand_data = None,
                           tracks_data=merger.tracks_data,
                           frames_data=merger.frames_data,
                           #    None,
                        #    chain_tracks=right_nodes,
                            chain_tracks=left_nodes,
                           shift_nframes=merger.START_TRACKING_TIME_SECONDS * \
                           merger.FRAME_RATE if merger.START_TRACKING_TIME_SECONDS else 0,
                           fix_text='2')
