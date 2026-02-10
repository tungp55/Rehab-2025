import os
import json
from utils import frame_data, box
import json
import os
import shutil


def read_file(fpath):
    f = open(fpath)
    data = json.load(f)
    fname = os.path.basename(fpath)
    str_fname = fname.replace('.json', '')
    splits = str_fname.split('_')
    frame_id = int(splits[1]) + 1
    frames = []
    for shape in data.get('shapes'):
        x1 = int(abs(shape.get('points')[0][0]))
        y1 = int(abs(shape.get('points')[0][1]))
        x2 = int(abs(shape.get('points')[1][0]))
        y2 = int(abs(shape.get('points')[1][1]))
        xmin = min(x1, x2)
        ymin = min(y1, y2)
        xmax = max(x1, x2)
        ymax = max(y1, y2)
        bounding_box = box(xmin, ymin, xmax - xmin, ymax - ymin)
        frame = frame_data(frame_id, int(shape.get('label')), bounding_box)
        frames.append(frame)
    f.close()
    return frames


def convert_json_2_tracks_file(input_dir='../tracking reindex/Cong Hai/GH010371_annotaions',
                               output_file='./GH010371_GT.txt'):
    files = os.listdir(input_dir)
    frames_data = []
    for file in files:
        file_path = os.path.join(input_dir, file)
        frames = read_file(file_path)
        frames_data.extend(frames)
    writer = open(output_file, "w+")
    for f_data in frames_data:
        line_string = "{}, {}, {}, {}, {}, {}\n".format(f_data.frame_id, f_data.track_id, f_data.bounding_box.x1, f_data.bounding_box.y1,
                                                        f_data.bounding_box.x2 - f_data.bounding_box.x1, f_data.bounding_box.y2 - f_data.bounding_box.y1)
        writer.write(line_string)
    writer.close()


def convert_tracks_file_2_json(text_file_path, output_dir,
                               output_dir_name,
                               output_prefix,
                               nframes=3600,
                               image_height=1440, image_width=1920):
    # path_images, txt_path, h, w = args.path_images, args.txt_path, args.height, args.width
    a = []
    count = 0
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    else:
        os.mkdir(output_dir)
    content = {"version": "4.5.9", "flags": {}}
    f = open(text_file_path, 'r')
    f_data = f.readlines()
    for i in range(0, nframes):
        fname = "{}{:05d}.json".format(output_prefix, i)
        file_path = os.path.join(output_dir, fname)
        json_file = open(file_path, 'w')
        json.dump(content, json_file)
        json_file.close()
        for line in f_data:
            data = [x for x in line.split(", ")]
            frame_id = int(data[0])
            if i == frame_id - 1:
                label = int(float((data[1])))
                x_min = abs(round(int(float(data[2]))))
                y_min = abs(round(int(float(data[3]))))
                x_max = x_min + abs(round(int(float(data[4]))))
                y_max = y_min + abs(round(int(float(data[5]))))
                dictionary = {
                    "label": f'{label}',
                    "points": [
                        [
                            x_min,
                            y_max
                        ],
                        [
                            x_max,
                            y_min
                        ]
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
                a.append(dictionary)
            else:
                continue
        json_file = open(file_path, 'r')
        json_data = json.load(json_file)
        json_file.close()
        json_data['version'] = "4.5.6"
        json_data['flags'] = {}
        json_data['shapes'] = a
        json_data["imagePath"] = "../{}/{}{:05d}.jpg".format(
            output_dir_name, output_prefix, i)
        json_data["imageData"] = None
        json_data["imageHeight"] = image_height
        json_data["imageWidth"] = image_width
        json_file = open(file_path, "w")
        json.dump(json_data, json_file)
        json_file.close()
        a = []
        f.close()


if __name__ == '__main__':
    # convert_tracks_file_2_json('F:/MITICA/merge tracks/GH010371_170_290_tay_bac_sy/merge_tracks_2_hands.txt',
    #                            'F:/MITICA/merge tracks/GH010371_170_290_tay_bac_sy/GH010371_merge_tracks_2_hands/',
    #                            'GH010371_')
    convert_tracks_file_2_json('J:/MITICA/merge tracks/GH010371_170_290_tay_bac_sy/merge_tracks_2_hands.txt',
                               'J:/MITICA/merge tracks/GH010371_170_290_tay_bac_sy/merge_tracks/',
                               'GH010371_170_290',
                               'GH010371_')
