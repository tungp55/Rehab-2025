import cv2
import json
import os
import argparse
import shutil

null = None
a = []
count = 0

parse = argparse.ArgumentParser()
parse.add_argument("-path_images", type=str, help="path to images")
# example: "F:/Database/Hand/GH_010373_380_500/GH010373_380_500"
parse.add_argument("-txt_path", type=str, help="path to txt track")
parse.add_argument("-height", type=int, default=1440, help="height_image")
parse.add_argument("-width", type=int, default=1920, help="width_image")
args = parse.parse_args()

if __name__ == "__main__":
    path_images, txt_path, h, w = args.path_images, args.txt_path, args.height, args.width
    list_images = os.listdir(path_images)
    number_images = len(list_images)
    path = path_images + "_annotations_ds/"
    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)
    content = {"version": "4.5.9", "flags": {}}
    for i in range(0, number_images):
        file_path = path + "GH010373_{:05d}.json".format(i)
        json_file = open(file_path, 'w')
        json.dump(content, json_file)
        json_file.close()
        f = open(txt_path, 'r')
        f_data = f.readlines()
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
                    "group_id": null,
                    "shape_type": "rectangle",
                    "flags": {}
                }
                a.append(dictionary)
            else:
                continue
        json_file = open(file_path, 'r')
        json_data = json.load(json_file)
        json_file.close()
        f.close()
        json_data['shapes'] = a
        json_data["imagePath"] = path_images + "\\GH010373_{:05d}.jpg".format(i)
        json_data["imageData"] = null
        json_data["imageHeight"] = h
        json_data["imageWidth"] = w
        json_file = open(file_path, "w")
        json.dump(json_data, json_file)
        json_file.close()
        a = []
