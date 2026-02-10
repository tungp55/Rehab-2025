import numpy as np
import cv2

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1440


def plotPoints(timecodes, bottom_line, line_height, frame_width, frame_rate, nr_of_frames):
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


def get_pointer_x(elapsed_time, frame_width, nr_of_frames, frame_rate):
    pointer_x = int(elapsed_time * frame_width / (nr_of_frames / frame_rate))
    return pointer_x


def triangle_pointers(x, y):
    pts = [[x, y],
           [x - 7, y + 7],
           [x + 7, y + 7]]
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    return pts


def draw_text(img, text,
              mode='auto',
              font=cv2.FONT_HERSHEY_PLAIN,
              bounding_box=(0, 0, 10, 10), resolution_rate=0.5,
              font_scale=3,
              font_thickness=2,
              text_color=(255, 255, 0),
              text_color_bg=(0, 0, 0)
              ):
    pt1 = (int(bounding_box.x1 * resolution_rate),
           int(bounding_box.y1 * resolution_rate))
    pt2 = (int(bounding_box.x2 * resolution_rate),
           int(bounding_box.y2 * resolution_rate))
    y_offset = 5
    if (pt1[0] < 10):
        x = pt2[0]
    else:
        x = pt1[0]
    if (pt1[1] < 10):
        y = pt2[1]
    else:
        y = pt1[1]
    y = y - y_offset
    if mode == 'center':
        x = int(FRAME_WIDTH * resolution_rate / 2)
        y = int(FRAME_HEIGHT * resolution_rate / 2)
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x, y), (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1),
                font, font_scale, text_color, lineType=1, thickness=2)
    return text_size


def draw_text_on_background(img, text,
                            font=cv2.FONT_HERSHEY_PLAIN,
                            point=(0, 0), 
                            font_scale=3,
                            font_thickness=2,
                            text_color=(255, 255, 0),
                            text_color_bg=(0, 0, 0)
                            ):
    x, y = point
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x, y), (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1),
                font, font_scale, text_color, lineType=1, thickness=2)
    return text_size


def visualize_chain_tracks(mode,
                           video_fname, output_fname, resolution_rate, acc_action_hand_data,
                           tracks_data, frames_data, chain_tracks, shift_nframes=0,
                           fix_text=None):
    # OUTPUT_FILE = 'J:/MITICA/reindex/GH010373_260_380_vung_nhanh/deep_sort_tracking.MP4'
    OUTPUT_FILE = output_fname
    BOX_COLOR = (255, 128, 0)
    ACTION_HAND_COLOR = (255, 0, 0)
    BOTTOM_LINE_Y = 50
    LINE_HEIGHT = 10
    POINTER_COLOR = (0, 0, 255)
    video = cv2.VideoCapture(video_fname)
    cv2.namedWindow("Video")
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    nr_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(
        *'MP4V'), frame_rate, (frame_width, frame_height))
    # action_hand_pts = plotPoints(
    #     acc_action_hand_data, BOTTOM_LINE_Y, LINE_HEIGHT, frame_width, frame_rate, nr_of_frames)
    while 1:
        current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
        elapsed_time_seconds = current_frame / frame_rate
        # Get the next videoframe
        ret, frame = video.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = int(4 * resolution_rate)
        font_text_color = (0, 255, 255)
        font_bgr_color = (0, 204, 102)
        font_thickness = int(4 * resolution_rate)
        line_type = 1
        # draw tracking bounding box
        if mode.startswith('deep_sort'):
            track_data_org = [
                frame for frame in frames_data if frame.frame_id == current_frame + shift_nframes + 1]
            for tracking_gt_frame in track_data_org:
                tracking_id = tracking_gt_frame.track_id
                bounding_box = tracking_gt_frame.bounding_box
                pt1 = (int(bounding_box.x1 * resolution_rate),
                       int(bounding_box.y1 * resolution_rate))
                pt2 = (int(bounding_box.x2 * resolution_rate),
                       int(bounding_box.y2 * resolution_rate))
                cv2.rectangle(frame, pt1, pt2,
                              BOX_COLOR, thickness=5)
                if mode == 'deep_sort':
                    draw_text(frame, str(tracking_id),
                              'auto',
                              font, bounding_box, resolution_rate,
                              font_scale, font_thickness, font_text_color, font_bgr_color)
                elif mode.endswith('center'):
                    draw_text(frame, str(tracking_id),
                              'center',
                              font, bounding_box, resolution_rate,
                              font_scale, font_thickness, font_text_color, font_bgr_color)

                    # cv2.putText(frame, str(tracking_id),
                    #             (pt1[0], pt1[1] - 5),
                    #             font,
                    #             1.5,
                    #             POINTER_COLOR,
                    #             lineType=1, thickness=2)
                    # draw merge_tracks result
        elif mode.startswith('merge_tracks'):
            # cv2.polylines(frame, [action_hand_pts],
            #               isClosed=False, color=ACTION_HAND_COLOR)
            tracks_by_frame = [track_id for track_id in chain_tracks
                               if tracks_data[track_id].start_frame <= current_frame + 1 + shift_nframes
                               and tracks_data[track_id].stop_frame >= current_frame + 1 + shift_nframes]
            if tracks_by_frame:
                track_id = tracks_by_frame[0]
                frame_data = [frame for frame in frames_data if frame.frame_id ==
                              current_frame + shift_nframes and frame.track_id == track_id]
                if frame_data:
                    bounding_box = frame_data[0].bounding_box
                    pt1 = (int(bounding_box.x1 * resolution_rate),
                           int(bounding_box.y1 * resolution_rate))
                    pt2 = (int(bounding_box.x2 * resolution_rate),
                           int(bounding_box.y2 * resolution_rate))
                    cv2.rectangle(frame, pt1, pt2,
                                  BOX_COLOR, thickness=5)
                    draw_text(frame,
                              fix_text if fix_text else str(track_id),
                              'center' if mode.endswith('center') else 'auto',
                              font, bounding_box, resolution_rate,
                              font_scale, font_thickness, font_text_color, font_bgr_color)
                    # draw diagonalize and track id
                    # cv2.line(frame, pt1, pt2, BOX_COLOR, thickness=2)
                    # cv2.putText(frame, str(track_id),
                    #             (int(frame_width / 2), int(frame_height / 2)),
                    #             font,
                    #             0.7,
                    #             BOX_COLOR,
                    #             lineType=1, thickness=2)
                    # draw acc line
                    # cv2.polylines(frame, [action_hand_pts],
                    #               isClosed=False, color=ACTION_HAND_COLOR)
                    # draw pointer
                    # pointer_x = get_pointer_x(
                    #     elapsed_time_seconds, frame_width, nr_of_frames, frame_rate)
                    # pts = triangle_pointers(pointer_x, BOTTOM_LINE_Y)
                    # cv2.fillPoly(frame, [pts], POINTER_COLOR)
        # show frame, break the loop if no frame is found
        if ret:
            cv2.imshow("Video", frame)
            out.write(frame)
            # update slider position on trackbar
            # NOTE: this is an expensive operation, remove to greatly increase max playback speed
            # cv2.setTrackbarPos("Frame", "Video", int(
            #     video.get(cv2.CAP_PROP_POS_FRAMES)))
        else:
            break
        key = cv2.waitKey(30)
        # stop playback when q is pressed
        if key == ord('q'):
            break
    # release resources
    video.release()
    out.release()
    cv2.destroyAllWindows()


def combine_3_videos(video_paths, output_path):
    w = 1920
    h = 1440
    blank_distance = 50
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = int(2)
    font_text_color = (0, 255, 255)
    font_bgr_color = (0, 204, 102)
    font_thickness = int(2)
    shift_y_text = 10
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
        *'MP4V'), 30, (w+w+blank_distance, h+h+blank_distance))
    cv2.namedWindow("Video")
    videos = []
    for video_fname in video_paths:
        videos.append(cv2.VideoCapture(video_fname))
    while 1:
        input_frames = []
        return_values = []
        for video in videos:
            ret, frame = video.read()
            input_frames.append(frame)
            return_values.append(ret)

        if len(input_frames) == 3 and return_values[0] and return_values[1] and return_values[2]:

            output_frame = np.zeros(
                (h + h + blank_distance, w+w+blank_distance,  3), dtype="uint8")
            frame0_start_x = int((w+w+blank_distance) / 2 - w / 2)
            draw_text_on_background(input_frames[0],'deep sort',
                                    font, (0, shift_y_text),
                                    font_scale, font_thickness, font_text_color, font_bgr_color)
            draw_text_on_background(input_frames[1],'left hand',
                                    font, (0, shift_y_text),
                                    font_scale, font_thickness, font_text_color, font_bgr_color)
            draw_text_on_background(input_frames[2],'right hand',
                                    font, (0, shift_y_text),
                                    font_scale, font_thickness, font_text_color, font_bgr_color)
            output_frame[0:h,
                         frame0_start_x: frame0_start_x + w] = input_frames[0]
            output_frame[h+blank_distance: h+h + blank_distance,
                         0:w] = input_frames[1]
            output_frame[h+blank_distance: h+h+blank_distance,
                         w + blank_distance:w+w+blank_distance] = input_frames[2]

            cv2.imshow("Video", output_frame)
            out.write(output_frame)
        else:
            break
        key = cv2.waitKey(30)
        # stop playback when q is pressed
        if key == ord('q'):
            break
    # release resources
    for video in videos:
        video.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    combine_3_videos(['J:/MITICA/merge tracks/GH010373_260_380_vung_nhanh/deep_sort.MP4',
                      'J:/MITICA/merge tracks/GH010373_260_380_vung_nhanh/left_hand.MP4',
                      'J:/MITICA/merge tracks/GH010373_260_380_vung_nhanh/right_hand.MP4'],
                     'J:/MITICA/merge tracks/GH010373_260_380_vung_nhanh/compare_cv2_v2.MP4')
