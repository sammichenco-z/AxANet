import os
import sys

from IPython import embed
import cv2
import random
import numpy as np
import time
from tqdm.autonotebook import tqdm

import pandas as pd
from openpyxl import load_workbook

LEN_CONCAT_VIDEO = 150
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
Freq = 10
time_frame_delta = 1 / Freq
single_time = 3 # second


def read_info_from_excecl(raw_data_excel_path):
    print('read excel...')
    df = pd.read_excel(raw_data_excel_path)
    print('read excel Done')

    

    try:
        raw_img_path = df['DRAMATestImage']
        X_coords = df['XTrackedTest']
        Y_coords = df['YTrackedTest']
    except:
        raw_img_path = df['DRAMATrainImage']
        X_coords = df['XTrackedIntro']
        Y_coords = df['YTrackedIntro']

    raw_paths = [i.split("\\")[-1] for i in raw_img_path]
    new_X_coords = []
    new_Y_coords = []
    new_img_paths = []
    for id_, i in enumerate(raw_img_path):
        if i.find("drama_random pic") == -1:
            continue
        tmp_path = i.split("\\")[-1].replace("_raw", "")
        if tmp_path.find("1titan") != -1:
            tmp_path = tmp_path.replace("1titan_", "titan/").replace("_frame_", "/frame_")
        else:
            tmp_path = tmp_path.replace("_frame_", "*").replace("_", "/").replace("*", "/frame_")
        new_img_paths.append(tmp_path.replace(".png", ""))
        new_X_coords.append(int(X_coords[id_] - 1920))
        new_Y_coords.append(int(Y_coords[id_]))

    return raw_paths, new_img_paths, new_X_coords, new_Y_coords


def get_concat_video_beginend_index(event):
    VideoStimulusStart_idx_list = []
    VideoStimulusEnd_idx_list = []
    for i in range(len(event)):
        if event[i] == 'VideoStimulusStart':
            VideoStimulusStart_idx_list.append(i)
        elif event[i] == 'VideoStimulusEnd':
            VideoStimulusEnd_idx_list.append(i)
    assert len(VideoStimulusStart_idx_list) == len(VideoStimulusEnd_idx_list)
    return VideoStimulusStart_idx_list, VideoStimulusEnd_idx_list


def get_gaze_point_4_ori(item_gaze_point_x, item_gaze_point_y, \
        item_ori_w, item_ori_h, item_pre_w, item_pre_h, item_pre_x, item_pre_y):
    image_point_x = item_gaze_point_x - item_pre_x
    image_point_y = item_gaze_point_y - item_pre_y

    item_gaze_point_x_ori = image_point_x / item_pre_w * item_ori_w
    item_gaze_point_y_ori = image_point_y / item_pre_h * item_ori_h

    return item_gaze_point_x_ori, item_gaze_point_y_ori


def get_gaze_point_4_concat_video_frame(all_ori_data, recording_timestamp, begin_index, end_index, begin_timestamp):
    eye_movement, gaze_point_x, gaze_point_y, ori_w, ori_h, pre_w, pre_h, pre_x, pre_y = all_ori_data

    gaze_point_4_frame = {}
    with tqdm(total=end_index-(begin_index+1)) as pbar:
        for gaze_i in range(begin_index + 1, end_index):
            if eye_movement[gaze_i] != 'Fixation':
                continue
            
            item_ori_w = ori_w[gaze_i]
            item_ori_h = ori_h[gaze_i]
            item_pre_w = pre_w[gaze_i]
            item_pre_h = pre_h[gaze_i]
            item_pre_x = pre_x[gaze_i]
            item_pre_y = pre_y[gaze_i]

            item_gaze_point_x = gaze_point_x[gaze_i]
            item_gaze_point_y = gaze_point_y[gaze_i]

            if np.isnan(item_gaze_point_x):
                continue

            item_gaze_point_x_ori, item_gaze_point_y_ori = get_gaze_point_4_ori(item_gaze_point_x, item_gaze_point_y, \
                    item_ori_w, item_ori_h, item_pre_w, item_pre_h, item_pre_x, item_pre_y)

            timestamp = recording_timestamp[gaze_i]
            rel_timestamp = timestamp - begin_timestamp
            rel_second = rel_timestamp / 1000
            rel_frame_idx = int(rel_second / time_frame_delta)

            if rel_frame_idx not in gaze_point_4_frame.keys():
                gaze_point_4_frame[rel_frame_idx] = []
            gaze_point_4_frame[rel_frame_idx].append((item_gaze_point_x_ori, item_gaze_point_y_ori))

            pbar.update(1)
    return gaze_point_4_frame


def read_video_info(video_info_path):
    with open(video_info_path) as f:
        video_info_list = [item.split('\n')[0] for item in f.readlines()]

    # TODO: prior knowledge. a concatenated video consists of LEN_CONCAT_VIDEO single videos. The last one has less than LEN_CONCAT_VIDEO videos.
    # assert len(video_info_list) == LEN_CONCAT_VIDEO 
    
    return video_info_list


def is_cross_label(frame):
    if frame.mean() > 245:  # TODO incomplete
        return True
    else:
        return False


def get_video_image_size(video_path):
    vidcap = cv2.VideoCapture(video_path)
    _, frame = vidcap.read()
    img_size = frame.shape[:2]  # height * width
    vidcap.release()
    return img_size


def get_frame_mapping(video_path, video_info_list, drama_path):

    ## get dir name
    outdir_name_list = []
    valid_or_not_list = []
    cross_frame_list = []
    for i in range(len(video_info_list)):
        video_info = video_info_list[i]

        outdir_1 = video_info.split('/')[-3]
        outdir_2 = video_info.split('/')[-2]
        cross_time = float(video_info.split('_')[-1][:-4])
        valid_or_not = video_info.split('_')[-2]
        cross_frame = int(cross_time * Freq)

        outdir_name_list.append(outdir_1+'/'+outdir_2)
        valid_or_not_list.append(valid_or_not)
        cross_frame_list.append(cross_frame)

    ## get drama frame name
    dir_frame_name_dic = {}
    for i in range(len(outdir_name_list)):
        outdir_name = outdir_name_list[i]
        outdir_name_complete = os.path.join(drama_path, outdir_name)
        files = sorted([os.path.join(outdir_name, f) for f in os.listdir(outdir_name_complete) if f.startswith('frame')])
        dir_frame_name_dic[outdir_name] = files

    ## get mapping relation
    videoframe_to_dramacombinedframe = {}
    dramacombinedframe_to_videoframe = {}
    single_video_cross_count = {}
    single_video_frame_count = {}

    vidcap = cv2.VideoCapture(video_path)
    frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    single_video_index = -1
    frame_index_in_video = -1
    cross_label_ing = False
    last_frame = 0
    with tqdm(total=int(frames)) as pbar:
        for idx in range(int(frames)):
            ## read frame
            _, frame = vidcap.read()

            ## count single video
            if is_cross_label(frame):
                if not cross_label_ing: # a new single video
                    cross_label_ing = True
                    single_video_index += 1
                    frame_index_in_video = -1
            else:
                cross_label_ing = False
            assert single_video_index < LEN_CONCAT_VIDEO

            ## count frame in a video
            if not cross_label_ing:
                frame_index_in_video += 1

            ## record mapping relation
            single_video_name = outdir_name_list[single_video_index]
            if cross_label_ing:
                videoframe_to_dramacombinedframe[idx] = None    # for cross, mapping to None
                if single_video_name not in single_video_cross_count.keys():
                    single_video_cross_count[single_video_name] = 0
                single_video_cross_count[single_video_name] += 1
            else:
                frame_single_video_name_list = dir_frame_name_dic[single_video_name]
                repeat_count = int(Freq * single_time / len(frame_single_video_name_list))
                corresponding_frame_name = frame_single_video_name_list[(frame_index_in_video // repeat_count)]

                videoframe_to_dramacombinedframe[idx] = corresponding_frame_name
                if corresponding_frame_name not in dramacombinedframe_to_videoframe.keys():
                    dramacombinedframe_to_videoframe[corresponding_frame_name] = []
                dramacombinedframe_to_videoframe[corresponding_frame_name].append(idx)

                if single_video_name not in single_video_frame_count.keys():
                    single_video_frame_count[single_video_name] = 0
                single_video_frame_count[single_video_name] += 1

            last_frame = frame

            pbar.update(1)

    vidcap.release()

    ## validation
    for k in dramacombinedframe_to_videoframe.keys():
        frame_length = len(dramacombinedframe_to_videoframe[k])
        assert frame_length >= 3
        if frame_length > 3:
            dir_name = os.path.join(k.split('/')[0], k.split('/')[1])
            assert len(dir_frame_name_dic[dir_name]) < 10
    for i in range(len(list(single_video_cross_count.keys()))):
        assert single_video_cross_count[list(single_video_cross_count.keys())[i]] == cross_frame_list[i]
    for i in range(len(list(single_video_frame_count.keys()))):
        count_0 = single_video_frame_count[list(single_video_frame_count.keys())[i]]
        count_1 = len(dir_frame_name_dic[list(dir_frame_name_dic.keys())[i]])
        assert count_0 == int(Freq * single_time / count_1) * count_1

    return videoframe_to_dramacombinedframe, dramacombinedframe_to_videoframe


def draw_gaze(gaze_point_4_frame, concat_video_dir, video_name, output_dir, drama_path):
    min_frame = min(gaze_point_4_frame.keys())
    max_frame = max(gaze_point_4_frame.keys())

    video_path = os.path.join(concat_video_dir, video_name)
    video_info_path = os.path.join(concat_video_dir, video_name.split('.')[0]+'.txt')
    video_info_list = read_video_info(video_info_path)

    ## get img_size
    img_size = get_video_image_size(video_path)

    ## get frame mapping relation
    videoframe_to_dramacombinedframe, dramacombinedframe_to_videoframe = get_frame_mapping(video_path, video_info_list, drama_path)

    ## draw
    ########################### gaussian kernel
    sigma = img_size[0] // 50
    kernel_size = 3 * sigma * 2 +1
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel_2D = gaussian_kernel @ gaussian_kernel.transpose()
    ###########################

    draw_concat_video_gaze_heatmap = False
    if draw_concat_video_gaze_heatmap:
        os.makedirs(os.path.join(output_dir, 'concat_video_gaze_heatmap'), exist_ok=True)
        output_video_file = os.path.join(output_dir, 'concat_video_gaze_heatmap', video_name)
        out_video = cv2.VideoWriter(output_video_file, fourcc, Freq, (img_size[1],img_size[0]))

        vidcap = cv2.VideoCapture(video_path)
        frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        try:
            assert frames >= max_frame # TODO max_frame count from 0?
        except:
            embed(header='bug')

        with tqdm(total=int(frames)) as pbar:
            for idx in range(int(frames)):
                if idx > max_frame:
                    break
                _, frame = vidcap.read()
                
                ## final video
                gaze_map = np.zeros_like(frame).astype(float)
                for item_gaze_point_x_ori, item_gaze_point_y_ori in gaze_point_4_frame[idx]:
                    if int(item_gaze_point_y_ori) >= gaze_map.shape[0] or int(item_gaze_point_y_ori) < 0 or \
                        int(item_gaze_point_x_ori) >= gaze_map.shape[1] or int(item_gaze_point_x_ori) < 0:
                        continue
                    try:
                        gaze_map[int(item_gaze_point_y_ori), int(item_gaze_point_x_ori)] = 255
                    except:
                        embed(header='bug')

                gaze_map_blur = cv2.filter2D(gaze_map, -1, kernel_2D)
                gaze_map_blur_max = gaze_map_blur.max()
                gaze_map_blur = gaze_map_blur / gaze_map_blur_max * 255. if gaze_map_blur_max > 0 else gaze_map_blur
                gaze_map_blur = gaze_map_blur.astype(np.uint8)

                heatmap = cv2.applyColorMap(gaze_map_blur, cv2.COLORMAP_JET)
                gaze_heatmap = cv2.addWeighted(frame,0.5,heatmap,0.5,0)

                out_video.write(gaze_heatmap)

                pbar.update(1)

        vidcap.release()
        out_video.release()

    draw_per_frame_gaze = True
    if draw_per_frame_gaze:
        output_image_path = os.path.join(output_dir, 'images')
        os.makedirs(output_image_path, exist_ok=True)
        
        points_per_dramacombinedframe = {}
        for k in dramacombinedframe_to_videoframe.keys():
            points_per_dramacombinedframe[k] = []

        ## get point
        vidcap = cv2.VideoCapture(video_path)
        frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        try:
            print('frames: ', frames, 'max_frame: ', max_frame)
            assert frames >= max_frame # TODO max_frame count from 0?
        except:
            embed(header='bug')
        for idx in range(int(frames)):
            if idx > max_frame:
                break
            _, frame = vidcap.read()

            dramacombinedframe_index = videoframe_to_dramacombinedframe[idx]
            if dramacombinedframe_index == None:
                continue

            if idx in gaze_point_4_frame.keys():
                points_per_dramacombinedframe[dramacombinedframe_index].extend(gaze_point_4_frame[idx])
        vidcap.release()

        ## draw
        with tqdm(total=len(list(points_per_dramacombinedframe.keys()))) as pbar:
            for k in points_per_dramacombinedframe.keys():
                # if len(points_per_dramacombinedframe[k]) > 0:
                #     continue

                frame = cv2.imread(os.path.join(drama_path, k))

                gaze_map = np.zeros_like(frame).astype(float)

                for item_gaze_point_x_ori, item_gaze_point_y_ori in points_per_dramacombinedframe[k]:
                    if int(item_gaze_point_y_ori) >= gaze_map.shape[0] or int(item_gaze_point_y_ori) < 0 or \
                        int(item_gaze_point_x_ori) >= gaze_map.shape[1] or int(item_gaze_point_x_ori) < 0:
                        continue
                    try:
                        gaze_map[int(item_gaze_point_y_ori), int(item_gaze_point_x_ori)] = 255
                    except:
                        embed(header='bug')

                gaze_map_blur = cv2.filter2D(gaze_map, -1, kernel_2D)
                gaze_map_blur_max = gaze_map_blur.max()
                gaze_map_blur = gaze_map_blur / gaze_map_blur_max * 255. if gaze_map_blur_max > 0 else gaze_map_blur
                gaze_map_blur = gaze_map_blur.astype(np.uint8)

                heatmap = cv2.applyColorMap(gaze_map_blur, cv2.COLORMAP_JET)
                gaze_heatmap = cv2.addWeighted(frame,0.5,heatmap,0.5,0)

                subdir_name = os.path.join(k.split('/')[0], k.split('/')[1])
                fixation_output_dir = os.path.join(output_image_path, subdir_name, 'fixation')
                gaze_output_dir = os.path.join(output_image_path, subdir_name, 'gaze')
                heatmap_output_dir = os.path.join(output_image_path, subdir_name, 'heatmap')
                os.makedirs(fixation_output_dir, exist_ok=True)
                os.makedirs(gaze_output_dir, exist_ok=True)
                os.makedirs(heatmap_output_dir, exist_ok=True)

                cv2.imwrite(os.path.join(fixation_output_dir, k.split('/')[-1]), gaze_map.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 100])
                cv2.imwrite(os.path.join(gaze_output_dir, k.split('/')[-1]), gaze_map_blur, [cv2.IMWRITE_JPEG_QUALITY, 100])
                cv2.imwrite(os.path.join(heatmap_output_dir, k.split('/')[-1]), gaze_heatmap, [cv2.IMWRITE_JPEG_QUALITY, 100])

                pbar.update(1)

def read_csv(path):
    data = np.loadtxt(open(path,"rb"), delimiter=",", skiprows=1, usecols=[0,1,2]) 

    return data



def main():
    path_find_img_time = "collect_data/User3you_230614205641_triggle2.csv"
    root_img_path = "/DATA_EDS/zyp/jinbu/datasets/drama/combined/"


    img_time_data = read_csv(path_find_img_time)
    image_id = 0
    start_time = 0
    end_time = 0
    if_start = False
    all_time_info = []
    for row_id, row in tqdm(enumerate(img_time_data)):
        if row[2] == 3:
            start_time = row[1]
            if_start = True
        if if_start and row[2] == 6 or row_id == len(img_time_data)-1:
            end_time = row[1]
            if_start = False
            all_time_info.append((start_time, end_time))

    raw_paths, raw_img_paths, X_coords, Y_coords = \
            read_info_from_excecl('collect_data/test_data.xlsx')

    assert len(all_time_info) == len(raw_paths), (len(all_time_info), len(raw_paths))

    path_gaze_data = "collect_data/User3you_230614205641.csv"
    gaze_data = read_csv(path_gaze_data)

    video_size = (1920, 1080)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter("gaze_video.mp4",  fourcc, 250, video_size, True)


    this_img_id = 0
    try:
        for row_id, row in tqdm(enumerate(gaze_data)):
            if this_img_id > len(all_time_info)-1:
                break
            if row[2] < all_time_info[this_img_id][0]:
                continue
            if row[2] > all_time_info[this_img_id][1]:
                this_img_id += 1
                continue
            raw_img_path = raw_img_paths[this_img_id]
            X_coord = row[0] * 1920
            Y_coord = row[1] * 1080
            raw_img = cv2.imread(os.path.join(root_img_path, raw_img_path+".png"))
            raw_img = cv2.resize(raw_img, video_size)

            if X_coord > 0 and Y_coord > 0:
                cv2.circle(raw_img, (int(X_coord), int(Y_coord)), 10, (0, 0, 255), 10)

            video.write(raw_img)
        video.release()
    except:
        print("except")
        video.release()
    
    print("done")







if __name__ == '__main__':
    main()
