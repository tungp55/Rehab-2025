import os
import json
import numpy as np
import math
import itertools

def get_box(obj, boxes_list, l):
    for i in range(len(obj)):
        label = int(obj[i]['label'])
        bbox = obj[i]['points']
        x1 = bbox[0][0]
        y1 = bbox[0][1]
        x2 = bbox[1][0]
        y2 = bbox[1][1]
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)
        box = [label, x_min, y_max, x_max, y_min]
        l.append(label)
        boxes_list.append(box)
    return boxes_list, l


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[1], boxB[1])
    yA = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    yB = max(boxA[4], boxB[4])
    interArea = max(0, xB - xA + 1) * max(0, yA - yB + 1)
    boxAArea = (boxA[3] - boxA[1] + 1) * (boxA[2] - boxA[4] + 1)
    boxBArea = (boxB[3] - boxB[1] + 1) * (boxB[2] - boxB[4] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def center_box(box):
    x_center = box[1] + (box[3] - box[1]) / 2
    y_center = box[4] + (box[2] - box[4]) / 2
    center_point = [x_center, y_center]
    return center_point


def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def calc_conditions(gt_boxes, pred_boxes, iou_thresh, id_switches, a):
    gt_class_ids_ = np.zeros(len(gt_boxes))
    pred_class_ids_ = np.zeros(len(pred_boxes))
    TP, FP, FN = 0, 0, 0
    for i, gt_box in enumerate(gt_boxes):
        iou = []
        ious = []
        gt_label = gt_box[0]
        for j, pre_box in enumerate(pred_boxes):
            now_iou = bb_intersection_over_union(gt_box, pre_box)
            ious.append(now_iou)
            if now_iou >= iou_thresh:
                iou.append(now_iou)
                gt_class_ids_[i] = 1
                pred_class_ids_[j] = 1
        if len(ious) != 0:
            max_iou = np.argmax(ious)
            for k, pre_box in enumerate(pred_boxes):
                if k == max_iou:
                    track_label = pre_box[0]

            if gt_label != track_label and max(ious) >= iou_thresh:
                id_switches.append(gt_label)
                a.append(track_label)
            if len(iou) > 0:
                TP += 1
                FP += len(iou) - 1
    FN += np.count_nonzero(np.array(gt_class_ids_) == 0)
    FP += np.count_nonzero(np.array(pred_class_ids_) == 0)
    return TP, FP, FN, id_switches


def evaluate(gt_path, track_path, gt_track_id = None):
    GT = 0
    sum_FP = 0
    sum_FN = 0
    dis = 0
    num_match = 0
    id_switch = []
    gt_label_list = []
    track_label_list = []
    overlap = 0
    a = []
    gt_json_name_list = []
    for file in os.listdir(gt_path):
        if file.endswith(".json"):
            gt_json_name_list.append(file)

    track_json_name_list = []
    for file in os.listdir(track_path):
        if file.endswith(".json"):
            track_json_name_list.append(file)

    for i in range(len(gt_json_name_list)):
        #Bách sửa: tìm file trùng tên, ko so sánh theo index, vì có thể thiếu frame trong 2 tập
        gt_json_name = gt_json_name_list[i]
        found_json_names = [fname for fname in track_json_name_list if fname == gt_json_name]
        if (len(found_json_names) > 0):
            track_path_json = os.path.join(track_path, gt_json_name)
        else: track_path_json = None
        gt_boxes_list = []
        track_boxes_list = []
        gt_boxes_list1 = []
        track_boxes_list1 = []
        gt_label_list1 = []
        track_label_list1 = []
        gt_path_json = gt_path + gt_json_name_list[i]
        #Bách comment
        # track_path_json = track_path + track_json_name_list[i]

        gt_json_file = open(gt_path_json, 'r')
        gt_json_data = json.load(gt_json_file)

        if track_path_json:
            track_json_file = open(track_path_json, 'r')
            track_json_data = json.load(track_json_file)
            track_obj = track_json_data['shapes']
        else: 
            track_json_data=[]
            track_obj = []
        gt_obj = gt_json_data['shapes']
        #Bách thêm để tính metric cho gt_track_id
        if gt_track_id:
            gt_obj = [obj for obj in gt_obj if obj['label']==gt_track_id]

        GT += len(gt_obj)
        get_box(gt_obj, gt_boxes_list, l=gt_label_list)
        get_box(track_obj, track_boxes_list, l=track_label_list)
        get_box(gt_obj, gt_boxes_list1, l=gt_label_list1)
        get_box(track_obj, track_boxes_list1, l=track_label_list1)
        TP, FP, FN, id_switches = calc_conditions(
            gt_boxes_list, track_boxes_list, iou_thresh=0.5, id_switches=id_switch, a=a)
        # print(i)
        # print("tp :", TP)
        # print("fp :", FP)
        # print("fn :", FN)
        sum_FP += FP
        sum_FN += FN
        # print(gt_label_list1)
        gt_ml = 0
        track_ml = 0
        for gt_label in gt_label_list1:
            # print(gt_label)
            for track_label in track_label_list1:
                if track_label == gt_label:

                    box_gt = [x for x in gt_boxes_list1 if x[0] == gt_label]
                    box_track = [
                        y for y in track_boxes_list1 if y[0] == gt_label]
                    # print(box_gt)
                    # print(box_track)
                    overlap += 1 - \
                        bb_intersection_over_union(box_gt[0], box_track[0])

                    if len(box_track) != 0:
                        num_match += 1
                        continue

    id_switch = sorted(list(set(a)))
    # print(id_switch)
    # print(gt_label_list)
    # print(track_label_list)
    k = gt_label_list
    gt_label_list = list(set(gt_label_list))
    MT = ML = e = 0
    for id in gt_label_list:
        gt_ml = 0
        track_ml = 0
        for label in k:
            if label == id:
                gt_ml += 1

        for labels in track_label_list:
            if labels == id:
                track_ml += 1
        if gt_ml != 0:
            if track_ml / gt_ml >= 0.8:
                MT += 1
            elif track_ml / gt_ml <= 0.2:
                ML += 1

    id_switch = len(list(set(id_switch)))
    track_label_list = list(set(track_label_list))
    gt_label_list = list(set(gt_label_list))
    print("id_switch : ", id_switch)
    print("sum_FP : ", sum_FP)
    print("sum_FN : ", sum_FN)
    print("GT objects count: ", GT)
    print("max pred label:", max(track_label_list))
    # print(track_label_list)
    # print(len(gt_label_list))
    # print(len(track_label_list))
    MOT_A = round((1 - (sum_FN + sum_FP + id_switch) / GT) * 100, 2)
    if num_match > 0:
        MOT_P = round((overlap / num_match) * 100, 2)
    else:
        MOT_P = 0
    print("MOTA : ", MOT_A)
    print("MT : ", MT)
    print("ML : ", ML)
    print("overlap : ", overlap)
    print("num_match : ", num_match)
    print("MOTP :", MOT_P)

def get_gt_track_id_set(gt_dir):
    gt_track_id_set = set()
    for file in os.listdir(gt_dir):
        if file.endswith(".json"):
            gt_json_path = os.path.join(gt_dir, file)
            gt_json_file = open(gt_json_path, 'r')
            gt_json_data = json.load(gt_json_file)
            for shape in gt_json_data['shapes']:
                gt_track_id_set.add(shape['label'])
    return gt_track_id_set

if __name__ == '__main__':
    gt_path = "J:/MITICA/merge tracks/GH010373_260_380_vung_nhanh/GH010373_260_380_gt/"
    pred_path = "J:/MITICA/merge tracks/GH010373_260_380_vung_nhanh/deep_sort/"
    print('===========mean results=================')
    evaluate(gt_path, pred_path)

    #loop gt_track_id
    gt_track_id_set = get_gt_track_id_set(gt_path)
    for gt_track_id in gt_track_id_set:        
        print('===========track_id #{} results================='.format(gt_track_id))
        evaluate(gt_path, pred_path, gt_track_id)
