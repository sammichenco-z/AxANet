import pandas as pd
import numpy as np
import os
import json


def process_raw(filename, driver_id, gt_df):
    t0_t1_X_std_df = pd.DataFrame({'driver_id': [driver_id] * 90,
                                   'img_id': list(range(90)),
                                   'hazard_task': [-1] * 90,
                                   'turn_task': [-1] * 90,
                                   'anomaly_task': [-1] * 90})
    t0_t1_Y_std_df = pd.DataFrame({'driver_id': [driver_id] * 90,
                                   'img_id': list(range(90)),
                                   'hazard_task': [-1] * 90,
                                   'turn_task': [-1] * 90,
                                   'anomaly_task': [-1] * 90})
    t1_t2_X_std_df = pd.DataFrame({'driver_id': [driver_id] * 90,
                                   'img_id': list(range(90)),
                                   'hazard_task': [-1] * 90,
                                   'turn_task': [-1] * 90,
                                   'anomaly_task': [-1] * 90})
    t1_t2_Y_std_df = pd.DataFrame({'driver_id': [driver_id] * 90,
                                   'img_id': list(range(90)),
                                   'hazard_task': [-1] * 90,
                                   'turn_task': [-1] * 90,
                                   'anomaly_task': [-1] * 90})
    t2_end_X_std_df = pd.DataFrame({'driver_id': [driver_id] * 90,
                                    'img_id': list(range(90)),
                                    'hazard_task': [-1] * 90,
                                    'turn_task': [-1] * 90,
                                    'anomaly_task': [-1] * 90})
    t2_end_Y_std_df = pd.DataFrame({'driver_id': [driver_id] * 90,
                                    'img_id': list(range(90)),
                                    'hazard_task': [-1] * 90,
                                    'turn_task': [-1] * 90,
                                    'anomaly_task': [-1] * 90})
    t0_t1_unique_df = pd.DataFrame({'driver_id': [driver_id] * 90,
                                    'img_id': list(range(90)),
                                    'hazard_task': [-1] * 90,
                                    'turn_task': [-1] * 90,
                                    'anomaly_task': [-1] * 90})
    t1_t2_unique_df = pd.DataFrame({'driver_id': [driver_id] * 90,
                                    'img_id': list(range(90)),
                                    'hazard_task': [-1] * 90,
                                    'turn_task': [-1] * 90,
                                    'anomaly_task': [-1] * 90})
    t2_end_unique_df = pd.DataFrame({'driver_id': [driver_id] * 90,
                                     'img_id': list(range(90)),
                                     'hazard_task': [-1] * 90,
                                     'turn_task': [-1] * 90,
                                     'anomaly_task': [-1] * 90})
    t0_t1_fixation_df = pd.DataFrame({'driver_id': [driver_id] * 90,
                                      'img_id': list(range(90)),
                                      'hazard_task': [-1] * 90,
                                      'turn_task': [-1] * 90,
                                      'anomaly_task': [-1] * 90})
    t1_t2_fixation_df = pd.DataFrame({'driver_id': [driver_id] * 90,
                                      'img_id': list(range(90)),
                                      'hazard_task': [-1] * 90,
                                      'turn_task': [-1] * 90,
                                      'anomaly_task': [-1] * 90})
    t2_end_fixation_df = pd.DataFrame({'driver_id': [driver_id] * 90,
                                       'img_id': list(range(90)),
                                       'hazard_task': [-1] * 90,
                                       'turn_task': [-1] * 90,
                                       'anomaly_task': [-1] * 90})

    t0_t1_time_df = pd.DataFrame({'driver_id': [driver_id] * 90,
                                  'img_id': list(range(90)),
                                  'hazard_task': [-1] * 90,
                                  'turn_task': [-1] * 90,
                                  'anomaly_task': [-1] * 90})
    t1_t2_time_df = pd.DataFrame({'driver_id': [driver_id] * 90,
                                  'img_id': list(range(90)),
                                  'hazard_task': [-1] * 90,
                                  'turn_task': [-1] * 90,
                                  'anomaly_task': [-1] * 90})
    t2_end_time_df = pd.DataFrame({'driver_id': [driver_id] * 90,
                                   'img_id': list(range(90)),
                                   'hazard_task': [-1] * 90,
                                   'turn_task': [-1] * 90,
                                   'anomaly_task': [-1] * 90})

    t0_t1_percentage_df = pd.DataFrame({'driver_id': [driver_id] * 90,
                                        'img_id': list(range(90)),
                                        'hazard_task': [-1] * 90,
                                        'turn_task': [-1] * 90,
                                        'anomaly_task': [-1] * 90})
    t1_t2_percentage_df = pd.DataFrame({'driver_id': [driver_id] * 90,
                                        'img_id': list(range(90)),
                                        'hazard_task': [-1] * 90,
                                        'turn_task': [-1] * 90,
                                        'anomaly_task': [-1] * 90})
    t2_end_percentage_df = pd.DataFrame({'driver_id': [driver_id] * 90,
                                         'img_id': list(range(90)),
                                         'hazard_task': [-1] * 90,
                                         'turn_task': [-1] * 90,
                                         'anomaly_task': [-1] * 90})

    file = pd.read_csv(filename, dtype={15: str, 18: str, 19: str})
    start_index = -1
    # print(file)
    for (index, row) in file.iterrows():
        if pd.notnull(row['name']) and start_index == -1:
            start_index = index
        elif pd.isnull(row['name']) and start_index != -1:
            end_index = index
            #
            temp_name = os.path.basename(str(file['name'].loc[start_index]))
            if temp_name.startswith('itan'):
                temp_name = 't' + temp_name
            img_row = gt_df[gt_df['name'] == temp_name]
            # print("image index", index, "img row: ", img_row)
            img_id = img_row.index[0]

            # compute aoi region
            x1 = img_row['x1'].item() * 1920
            y1 = img_row['y1'].item() * 1080
            x2 = img_row['x2'].item() * 1920
            y2 = img_row['y2'].item() * 1080

            if 30 < img_id < 60:
                # special computation because the ground truth was record in resolution of 1280 * 720
                x1 = img_row['x1'].item() / 1280 * 1920
                y1 = img_row['y1'].item() / 720 * 1080
                x2 = img_row['x2'].item() / 1280 * 1920
                y2 = img_row['y2'].item() / 720 * 1080

            # get task id
            if file['task'].loc[start_index] == "Hazard":
                task = "Hazard"
            elif file['task'].loc[start_index] == "Turn":
                task = "Turn"
            elif file['task'].loc[start_index] == "Anomaly":
                task = "Anomaly"
            # initialise
            start_time = file["Recording Time Stamp[ms]"].loc[start_index]
            end_time = file["Recording Time Stamp[ms]"].loc[end_index - 1]
            new_fixation = False
            minor_x = -1
            minor_y = -1
            aoi_status = -1
            last_status = -1
            aoi_list = []

            for minor_index, minor_row in file.loc[start_index:(end_index - 1)].iterrows():
                if (minor_x != minor_row['Fixation Point X[px]'] or minor_y != minor_row['Fixation Point Y[px]']) and \
                        pd.notnull(minor_row['Fixation Point X[px]']) and pd.notnull(minor_row['Fixation Point Y[px]']):
                    # TODO new fixation
                    new_fixation = True
                    # compare using fixation points
                    minor_x = minor_row['Fixation Point X[px]']
                    minor_y = minor_row['Fixation Point Y[px]']

                if ((minor_x != minor_row['Fixation Point X[px]'] or minor_y != minor_row['Fixation Point Y[px]']) or
                    minor_index == (end_index - 1)) and new_fixation:
                    # end of fixation
                    new_fixation = False

                if new_fixation:

                    x = minor_row['Fixation Point X[px]']
                    y = minor_row['Fixation Point Y[px]']
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        aoi_status = 1
                    else:
                        aoi_status = 0
                    aoi_list.append([aoi_status, minor_row["Recording Time Stamp[ms]"], x, y, minor_index])

            # initialise
            in_aoi_time = 0
            in_aoi = 0
            t2_end = 0
            out_aoi = 0
            stay_aoi = 0
            firstaoi = True
            firstend = True
            firstendtime = -1
            sample_length = end_time - start_time
            first_start_index = -1
            first_end_index = -1
            later_aoi_start = 0

            in_aoi_flag = False
            for i in range(1, len(aoi_list)):
                last_status = aoi_list[i - 1][0]
                last_time = aoi_list[i - 1][1]
                last_x = aoi_list[i - 1][2]
                last_y = aoi_list[i - 1][3]
                current_status = aoi_list[i][0]
                current_time = aoi_list[i][1]
                current_x = aoi_list[i][2]
                current_y = aoi_list[i][3]

                if last_status == 0 and current_status == 1:
                    in_aoi += 1
                    in_aoi_flag = True
                    if firstaoi:
                        firstaoi = False
                        first_start_index = aoi_list[i][4]
                    else:
                        later_aoi_start = current_time

                elif last_status == 1 and current_status == 0:
                    out_aoi += 1
                    if firstaoi:
                        firstaoi = False
                        first_start_index = aoi_list[i - 1][4]
                    elif later_aoi_start == 0:
                        later_aoi_start = last_time
                    if firstend:
                        firstend = False
                        firstendtime = last_time
                        first_end_index = aoi_list[i - 1][4]
                    else:
                        t2_end += last_time - later_aoi_start
                        later_aoi_start = 0

                    in_aoi_flag = False
                elif last_status == 1 and current_status == 1:
                    if firstaoi:
                        firstaoi = False
                        first_start_index = aoi_list[i - 1][4]
                        # print("first aoi at", last_time)
                    if i == 1 and not in_aoi_flag:
                        in_aoi_flag = True
                        in_aoi += 1
                        firstaoi = False
                    else:
                        in_aoi_time += current_time - last_time
                        if last_x != current_x and last_y != current_y:
                            stay_aoi += 1

                if firstendtime == -1 and i == (len(aoi_list) - 1) and firstend and current_status == 1:
                    print("aoi countiue to the end")
                    firstendtime = current_time
                    first_end_index = aoi_list[i][4]
                    t2_end = 0  # The gaze does not leave the AOI until the end after first enter
                elif firstendtime != -1 and i == (len(aoi_list) - 1) and not firstend and current_status == 1 \
                        and later_aoi_start != 0:
                    t2_end += current_time - later_aoi_start

            if firstendtime != -1:
                t0_t1_df_m = file.iloc[start_index:first_start_index]
                t1_t2_df_m = file.iloc[first_start_index:first_end_index]
                t2_end_df_m = file.iloc[first_end_index:end_index]
                t0_t1_duration = file.iloc[first_start_index]['Recording Time Stamp[ms]'] - \
                                 file.iloc[start_index]['Recording Time Stamp[ms]']
                t1_t2_duration = file.iloc[first_end_index]['Recording Time Stamp[ms]'] - \
                                 file.iloc[first_start_index]['Recording Time Stamp[ms]']
                t2_end_duration = file.iloc[end_index]['Recording Time Stamp[ms]'] - \
                                  file.iloc[first_end_index]['Recording Time Stamp[ms]']

                # TODO count fixation time
                dfs = [t0_t1_df_m, t1_t2_df_m, t2_end_df_m]
                avg_durations = []

                for df_m in dfs:
                    df_copy = df_m.copy()  # Create a copy
                    if df_copy.empty:
                        print("DataFrame is empty，skip computation")
                        avg_durations.append(np.nan)
                        continue

                    # Delete any row contain Nan in column 'Fixation Point X[px]' and 'Fixation Point Y[px]'
                    df_copy.dropna(subset=['Fixation Point X[px]', 'Fixation Point Y[px]'], inplace=True)

                    df_copy.loc[:, 'fixation'] = (
                            (df_copy['Fixation Point X[px]'] != df_copy['Fixation Point X[px]'].shift(1)) |
                            (df_copy['Fixation Point Y[px]'] != df_copy['Fixation Point Y[px]'].shift(1)))

                    df_copy.loc[:, 'fixation_id'] = df_copy['fixation'].cumsum()

                    fixation_duration = df_copy.groupby('fixation_id')['Recording Time Stamp[ms]'].apply(
                        lambda x: x.iloc[-1] - x.iloc[0])

                    average_duration = fixation_duration.mean()
                    avg_durations.append(average_duration)

                t0_t1_fix_avg, t1_t2_fix_avg, t2_end_fix_avg = avg_durations

                print(
                    f"t0_t1_fix_avg: {t0_t1_fix_avg}, t1_t2_fix_avg: {t1_t2_fix_avg}, t2_end_fix_avg: {t2_end_fix_avg}")

                t0_t1_df_mm = t0_t1_df_m.drop_duplicates(subset=['Fixation Point X[px]', 'Fixation Point Y[px]'])
                t1_t2_df_mm = t1_t2_df_m.drop_duplicates(subset=['Fixation Point X[px]', 'Fixation Point Y[px]'])
                t2_end_df_mm = t2_end_df_m.drop_duplicates(subset=['Fixation Point X[px]', 'Fixation Point Y[px]'])

                t0_t1_unique_count = len(t0_t1_df_mm)
                t1_t2_unique_count = len(t1_t2_df_mm)
                t2_end_unique_count = len(t2_end_df_mm)

                # drop nan in
                t0_t1_df_m.dropna(subset=['Fixation Point X[px]', 'Fixation Point Y[px]'], inplace=True)
                t1_t2_df_m.dropna(subset=['Fixation Point X[px]', 'Fixation Point Y[px]'], inplace=True)
                t2_end_df_m.dropna(subset=['Fixation Point X[px]', 'Fixation Point Y[px]'], inplace=True)

                # compute the standard deviation
                t0_t1_X_std = t0_t1_df_m[['Gaze Point X[px]']].std()
                t0_t1_Y_std = t0_t1_df_m[['Gaze Point Y[px]']].std()
                t1_t2_X_std = t1_t2_df_m[['Gaze Point X[px]']].std()
                t1_t2_Y_std = t1_t2_df_m[['Gaze Point Y[px]']].std()
                t2_end_X_std = t2_end_df_m[['Gaze Point X[px]']].std()
                t2_end_Y_std = t2_end_df_m[['Gaze Point Y[px]']].std()

                if task == 'Hazard':
                    t0_t1_X_std_df.iloc[img_id, 2] = t0_t1_X_std
                    t0_t1_Y_std_df.iloc[img_id, 2] = t0_t1_Y_std
                    t1_t2_X_std_df.iloc[img_id, 2] = t1_t2_X_std
                    t1_t2_Y_std_df.iloc[img_id, 2] = t1_t2_Y_std
                    t2_end_X_std_df.iloc[img_id, 2] = t2_end_X_std
                    t2_end_Y_std_df.iloc[img_id, 2] = t2_end_Y_std
                    t0_t1_unique_df.iloc[img_id, 2] = t0_t1_unique_count
                    t1_t2_unique_df.iloc[img_id, 2] = t1_t2_unique_count
                    t2_end_unique_df.iloc[img_id, 2] = t2_end_unique_count
                    t0_t1_fixation_df.iloc[img_id, 2] = t0_t1_fix_avg
                    t1_t2_fixation_df.iloc[img_id, 2] = t1_t2_fix_avg
                    t2_end_fixation_df.iloc[img_id, 2] = t2_end_fix_avg
                    t0_t1_time_df.iloc[img_id, 2] = t0_t1_duration
                    t1_t2_time_df.iloc[img_id, 2] = t1_t2_duration
                    t2_end_time_df.iloc[img_id, 2] = t2_end_duration
                    t0_t1_percentage_df.iloc[img_id, 2] = t0_t1_duration / sample_length
                    t1_t2_percentage_df.iloc[img_id, 2] = t1_t2_duration / sample_length
                    t2_end_percentage_df.iloc[img_id, 2] = t2_end_duration / sample_length
                elif task == 'Turn':
                    t0_t1_X_std_df.iloc[img_id, 3] = t0_t1_X_std
                    t0_t1_Y_std_df.iloc[img_id, 3] = t0_t1_Y_std
                    t1_t2_X_std_df.iloc[img_id, 3] = t1_t2_X_std
                    t1_t2_Y_std_df.iloc[img_id, 3] = t1_t2_Y_std
                    t2_end_X_std_df.iloc[img_id, 3] = t2_end_X_std
                    t2_end_Y_std_df.iloc[img_id, 3] = t2_end_Y_std
                    t0_t1_unique_df.iloc[img_id, 3] = t0_t1_unique_count
                    t1_t2_unique_df.iloc[img_id, 3] = t1_t2_unique_count
                    t2_end_unique_df.iloc[img_id, 3] = t2_end_unique_count
                    t0_t1_fixation_df.iloc[img_id, 3] = t0_t1_fix_avg
                    t1_t2_fixation_df.iloc[img_id, 3] = t1_t2_fix_avg
                    t2_end_fixation_df.iloc[img_id, 3] = t2_end_fix_avg
                    t0_t1_time_df.iloc[img_id, 3] = t0_t1_duration
                    t1_t2_time_df.iloc[img_id, 3] = t1_t2_duration
                    t2_end_time_df.iloc[img_id, 3] = t2_end_duration
                    t0_t1_percentage_df.iloc[img_id, 3] = t0_t1_duration / sample_length
                    t1_t2_percentage_df.iloc[img_id, 3] = t1_t2_duration / sample_length
                    t2_end_percentage_df.iloc[img_id, 3] = t2_end_duration / sample_length
                elif task == 'Anomaly':
                    t0_t1_X_std_df.iloc[img_id, 4] = t0_t1_X_std
                    t0_t1_Y_std_df.iloc[img_id, 4] = t0_t1_Y_std
                    t1_t2_X_std_df.iloc[img_id, 4] = t1_t2_X_std
                    t1_t2_Y_std_df.iloc[img_id, 4] = t1_t2_Y_std
                    t2_end_X_std_df.iloc[img_id, 4] = t2_end_X_std
                    t2_end_Y_std_df.iloc[img_id, 4] = t2_end_Y_std
                    t0_t1_unique_df.iloc[img_id, 4] = t0_t1_unique_count
                    t1_t2_unique_df.iloc[img_id, 4] = t1_t2_unique_count
                    t2_end_unique_df.iloc[img_id, 4] = t2_end_unique_count
                    t0_t1_fixation_df.iloc[img_id, 4] = t0_t1_fix_avg
                    t1_t2_fixation_df.iloc[img_id, 4] = t1_t2_fix_avg
                    t2_end_fixation_df.iloc[img_id, 4] = t2_end_fix_avg
                    t0_t1_time_df.iloc[img_id, 4] = t0_t1_duration
                    t1_t2_time_df.iloc[img_id, 4] = t1_t2_duration
                    t2_end_time_df.iloc[img_id, 4] = t2_end_duration
                    t0_t1_percentage_df.iloc[img_id, 4] = t0_t1_duration / sample_length
                    t1_t2_percentage_df.iloc[img_id, 4] = t1_t2_duration / sample_length
                    t2_end_percentage_df.iloc[img_id, 4] = t2_end_duration / sample_length

            start_index = -1

    return t0_t1_df_m, t1_t2_df_m, t2_end_df_m, t0_t1_X_std_df, t0_t1_Y_std_df, t1_t2_X_std_df, \
        t1_t2_Y_std_df, t2_end_X_std_df, t2_end_Y_std_df, t0_t1_unique_df, t1_t2_unique_df, t2_end_unique_df, \
        t0_t1_fixation_df, t1_t2_fixation_df, t2_end_fixation_df, t0_t1_time_df, t1_t2_time_df, t2_end_time_df, \
        t0_t1_percentage_df, t1_t2_percentage_df, t2_end_percentage_df


# 定义一个函数，用于检查输出文件是否已经存在
def output_exists(output_path, file):
    # 定义输出文件名
    time_filename = os.path.join(output_path, file[:-4] + "_time.csv")

    # 检查输出文件是否都已经存在
    return os.path.exists(time_filename)


if __name__ == '__main__':

    img_list = pd.read_excel('M:\\DRAMA_data\\actual\\img_list.xlsx')
    # Extract the base names from the 'img_names' column
    base_names = img_list['img_names'].apply(os.path.basename)
    # Add a 't' to the beginning of each base name if it starts with 'itan'
    new_names = base_names.apply(lambda x: 't' + x if x.startswith('itan') else x)
    img_list['img_names'] = new_names
    # print("IMG_LIST \n", img_list)

    asee_path_new = "M:\\DRAMA_data\\actual\\processed_data\\new"
    asee_path_old = "M:\\DRAMA_data\\actual\\processed_data\\old"
    eprime_path = "M:\\DRAMA_data\\actual\\eprime_data"
    output_path = "M:\\DRAMA_data\\actual\\re-examine_script"

    # 获取asee_path中所有以.csv结尾的文件
    aseefiles_new = [file for file in os.listdir(asee_path_new) if file.endswith("fixation_only.csv")]
    aseefiles_old = [file for file in os.listdir(asee_path_old) if file.endswith("fixation_only.csv")]

    index = 0

    # initialise ground truth
    # Load the anomaly_GT JSON file
    with open("M:\\DRAMA_data\\actual\\anomaly_bbox_gt.json", 'r') as f:
        anomaly_GT = json.load(f)

    # Load the drama_GT JSON file
    with open("M:\\DRAMA_data\\actual\\drama_bbox_gt.json", 'r') as f:
        drama_GT = json.load(f)

    turn_GT = pd.read_excel("M:\\DRAMA_data\\actual\\TurnGroundTruth.xlsx")
    # print(turn_GT)

    gt_data = []
    index = 0
    # Iterate over the items in the JSON data
    for name, values in drama_GT.items():
        x1, y1, w, h = values
        x2 = x1 + w
        y2 = y1 + h
        gt_data.append([name, x1, y1, x2, y2])

    for (iindex, row) in turn_GT.iterrows():
        x1 = row["x"]
        x2 = x1 + row["w"]
        y1 = row["y"]
        y2 = y1 + row["h"]
        name = row["name"]
        gt_data.append([name, x1, y1, x2, y2])

    for name, values in anomaly_GT.items():
        # Get the bounding box information for the current object
        bbox = values['bbox']
        # Unpack the bounding box values into separate variables
        x1, y1, x2, y2 = bbox
        complete_name = name + ".jpeg"
        # Append a new row to the gt_data list
        gt_data.append([complete_name, x1, y1, x2, y2])

    gt_df = pd.DataFrame(gt_data, columns=['name', 'x1', 'y1', 'x2', 'y2'])

    # initialise storage data frame
    t0_t1_X_std = pd.DataFrame()
    t0_t1_Y_std = pd.DataFrame()
    t1_t2_X_std = pd.DataFrame()
    t1_t2_Y_std = pd.DataFrame()
    t2_end_X_std = pd.DataFrame()
    t2_end_Y_std = pd.DataFrame()
    t0_t1_unique = pd.DataFrame()
    t1_t2_unique = pd.DataFrame()
    t2_end_unique = pd.DataFrame()
    t0_t1_fixation = pd.DataFrame()
    t1_t2_fixation = pd.DataFrame()
    t2_end_fixation = pd.DataFrame()
    t0_t1_duration = pd.DataFrame()
    t1_t2_duration = pd.DataFrame()
    t2_end_duration = pd.DataFrame()
    t0_t1_percentage = pd.DataFrame()
    t1_t2_percentage = pd.DataFrame()
    t2_end_percentage = pd.DataFrame()

    index = 0
    for file in aseefiles_new:
        print("PROCESSING ", file)
        compplete_path = asee_path_new + "\\" + file
        index += 1

        t0_t1_df, t1_t2_df, t2_end_df, new_t0_t1_X_std, new_t0_t1_Y_std, new_t1_t2_X_std, new_t1_t2_Y_std, \
            new_t2_end_X_std, new_t2_end_Y_std, new_t0_t1_unique, new_t1_t2_unique, new_t2_end_unique, \
            new_t0_t1_fixation, new_t1_t2_fixation, new_t2_end_fixation, new_t0_t1_duration, new_t1_t2_duration, \
            new_t2_end_duration, new_t0_t1_percentage, new_t1_t2_percentage, new_t2_end_percentage = \
            process_raw(compplete_path, index, gt_df)

        t0_t1_X_std = pd.concat([t0_t1_X_std, new_t0_t1_X_std], ignore_index=True)
        t0_t1_Y_std = pd.concat([t0_t1_Y_std, new_t0_t1_Y_std], ignore_index=True)
        t1_t2_X_std = pd.concat([t1_t2_X_std, new_t1_t2_X_std], ignore_index=True)
        t1_t2_Y_std = pd.concat([t1_t2_Y_std, new_t1_t2_Y_std], ignore_index=True)
        t2_end_X_std = pd.concat([t2_end_X_std, new_t2_end_X_std], ignore_index=True)
        t2_end_Y_std = pd.concat([t2_end_Y_std, new_t2_end_Y_std], ignore_index=True)
        t0_t1_unique = pd.concat([t0_t1_unique, new_t0_t1_unique], ignore_index=True)
        t1_t2_unique = pd.concat([t1_t2_unique, new_t1_t2_unique], ignore_index=True)
        t2_end_unique = pd.concat([t2_end_unique, new_t2_end_unique], ignore_index=True)

        t0_t1_fixation = pd.concat([t0_t1_fixation, new_t0_t1_fixation], ignore_index=True)
        t1_t2_fixation = pd.concat([t1_t2_fixation, new_t1_t2_fixation], ignore_index=True)
        t2_end_fixation = pd.concat([t2_end_fixation, new_t2_end_fixation], ignore_index=True)

        t0_t1_duration = pd.concat([t0_t1_duration, new_t0_t1_duration], ignore_index=True)
        t1_t2_duration = pd.concat([t1_t2_duration, new_t1_t2_duration], ignore_index=True)
        t2_end_duration = pd.concat([t2_end_duration, new_t2_end_duration], ignore_index=True)

        t0_t1_percentage = pd.concat([t0_t1_percentage, new_t0_t1_percentage], ignore_index=True)
        t1_t2_percentage = pd.concat([t1_t2_percentage, new_t1_t2_percentage], ignore_index=True)
        t2_end_percentage = pd.concat([t2_end_percentage, new_t2_end_percentage], ignore_index=True)

        temp_name = os.path.basename(file)
        t1_t2_df.to_csv(output_path + "\\t1_t2_" + temp_name + ".csv")
        t0_t1_df.to_csv(output_path + "\\t0_t1_" + temp_name + ".csv")
        t2_end_df.to_csv(output_path + "\\t2_end_" + temp_name + ".csv")
    for file in aseefiles_old:
        print("PROCESSING ", file)
        print("\n driver_id: ", index)
        compplete_path = asee_path_old + "\\" + file
        index += 1

        t0_t1_df, t1_t2_df, t2_end_df, new_t0_t1_X_std, new_t0_t1_Y_std, new_t1_t2_X_std, new_t1_t2_Y_std, \
            new_t2_end_X_std, new_t2_end_Y_std, new_t0_t1_unique, new_t1_t2_unique, new_t2_end_unique, \
            new_t0_t1_fixation, new_t1_t2_fixation, new_t2_end_fixation, new_t0_t1_duration, new_t1_t2_duration, \
            new_t2_end_duration, new_t0_t1_percentage, new_t1_t2_percentage, new_t2_end_percentage = \
            process_raw(compplete_path, index, gt_df)

        t0_t1_X_std = pd.concat([t0_t1_X_std, new_t0_t1_X_std], ignore_index=True)
        t0_t1_Y_std = pd.concat([t0_t1_Y_std, new_t0_t1_Y_std], ignore_index=True)
        t1_t2_X_std = pd.concat([t1_t2_X_std, new_t1_t2_X_std], ignore_index=True)
        t1_t2_Y_std = pd.concat([t1_t2_Y_std, new_t1_t2_Y_std], ignore_index=True)
        t2_end_X_std = pd.concat([t2_end_X_std, new_t2_end_X_std], ignore_index=True)
        t2_end_Y_std = pd.concat([t2_end_Y_std, new_t2_end_Y_std], ignore_index=True)
        t0_t1_unique = pd.concat([t0_t1_unique, new_t0_t1_unique], ignore_index=True)
        t1_t2_unique = pd.concat([t1_t2_unique, new_t1_t2_unique], ignore_index=True)
        t2_end_unique = pd.concat([t2_end_unique, new_t2_end_unique], ignore_index=True)

        t0_t1_fixation = pd.concat([t0_t1_fixation, new_t0_t1_fixation], ignore_index=True)
        t1_t2_fixation = pd.concat([t1_t2_fixation, new_t1_t2_fixation], ignore_index=True)
        t2_end_fixation = pd.concat([t2_end_fixation, new_t2_end_fixation], ignore_index=True)

        t0_t1_duration = pd.concat([t0_t1_duration, new_t0_t1_duration], ignore_index=True)
        t1_t2_duration = pd.concat([t1_t2_duration, new_t1_t2_duration], ignore_index=True)
        t2_end_duration = pd.concat([t2_end_duration, new_t2_end_duration], ignore_index=True)

        t0_t1_percentage = pd.concat([t0_t1_percentage, new_t0_t1_percentage], ignore_index=True)
        t1_t2_percentage = pd.concat([t1_t2_percentage, new_t1_t2_percentage], ignore_index=True)
        t2_end_percentage = pd.concat([t2_end_percentage, new_t2_end_percentage], ignore_index=True)

        temp_name = os.path.basename(file)
        t1_t2_df.to_csv(output_path + "\\t1_t2_" + temp_name + ".csv")
        t0_t1_df.to_csv(output_path + "\\t0_t1_" + temp_name + ".csv")
        t2_end_df.to_csv(output_path + "\\t2_end_" + temp_name + ".csv")

    output_name = output_path + "\\t0_t1_X_std_fixationavg"
    t0_t1_X_std.to_csv(output_name + ".csv")
    output_name = output_path + "\\t0_t1_Y_std_fixationavg"
    t0_t1_Y_std.to_csv(output_name + ".csv")
    output_name = output_path + "\\t1_t2_X_std_fixationavg"
    t1_t2_X_std.to_csv(output_name + ".csv")
    output_name = output_path + "\\t1_t2_Y_std_fixationavg"
    t1_t2_Y_std.to_csv(output_name + ".csv")
    output_name = output_path + "\\t2_end_X_std_fixationavg"
    t2_end_X_std.to_csv(output_name + ".csv")
    output_name = output_path + "\\t2_end_Y_std_fixationavg"

    t2_end_Y_std.to_csv(output_name + ".csv")
    output_name = output_path + "\\t0_t1_unique_fixationavg"
    t0_t1_unique.to_csv(output_name + ".csv")
    output_name = output_path + "\\t1_t2_unique_fixationavg"
    t1_t2_unique.to_csv(output_name + ".csv")
    output_name = output_path + "\\t2_end_unique_fixationavg"
    t2_end_unique.to_csv(output_name + ".csv")

    output_name = output_path + "\\t0_t1_fix_avg_fixationavg"
    t0_t1_fixation.to_csv(output_name + ".csv")
    output_name = output_path + "\\t1_t2_fix_avg_fixationavg"
    t1_t2_fixation.to_csv(output_name + ".csv")
    output_name = output_path + "\\t2_end_fix_avg_fixationavg"
    t2_end_fixation.to_csv(output_name + ".csv")

    output_name = output_path + "\\t0_t1_duration_fixationavg"
    t0_t1_duration.to_csv(output_name + ".csv")
    output_name = output_path + "\\t1_t2_duration_fixationavg"
    t1_t2_duration.to_csv(output_name + ".csv")
    output_name = output_path + "\\t2_end_duration_fixationavg"
    t2_end_duration.to_csv(output_name + ".csv")

    output_name = output_path + "\\t0_t1_percentage_fixationavg"
    t0_t1_percentage.to_csv(output_name + ".csv")
    output_name = output_path + "\\t1_t2_percentage_fixationavg"
    t1_t2_percentage.to_csv(output_name + ".csv")
    output_name = output_path + "\\t2_end_percentage_fixationavg"
    t2_end_percentage.to_csv(output_name + ".csv")
