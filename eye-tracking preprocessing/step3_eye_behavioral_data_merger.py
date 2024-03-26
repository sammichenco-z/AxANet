import pandas as pd
import numpy as np
import os
import re

def process_raw(asee_filename, folder_path, files):
    exel_list = []
    stat_data = pd.DataFrame(index=range(90), columns=['img_id', 'Hazard', 'Turn', 'Anomaly', 'img_name'])
    gt_list = pd.read_csv("M:\\DRAMA_data\\actual\\gtlist.csv")
    stat_data['img_id'] = gt_list['index']
    last_order = 0
    count_order = ""
    print("before ", files)
    files.sort(key=lambda x: int(x.split("-")[1]))
    print("after ", files)
    for file in files:
        print(file)
        if file.endswith(".xlsx"):
            exel = pd.read_excel(os.path.join(folder_path, file))
            exel["name"] = np.nan
            exel['x'] = np.nan
            exel['y'] = np.nan
            exel['respond'] = np.nan
            exel['task'] = np.nan
            # Drop the second to sixth row in the xlsx files
            exel = exel.drop(exel.index[[1, 2, 3, 4, 5]])
            # Store the column "TurnTestImage" as "name" and the column "TurnTestDisplay.RESP" according to the
            # TurnTestImage column for xlsx files whose filename ends with "Turn"
            if file[23:-14].endswith("Turn"):
                exel["name"] = exel["TurnTestImage"]
                exel["respond"] = exel["TurnTestDisplay.RESP"]
                exel["task"] = "Turn"
                count_order += "2"
                print("File processed ", file)
            elif file[23:-14].endswith("Hazard"):
                exel["name"] = exel["DRAMATestImage"]
                exel["x"] = exel["XTrackedTest"]
                exel["y"] = exel["YTrackedTest"]
                exel["task"] = "Hazard"
                count_order += "1"
                print("File processed ", file)
            elif file[23:-14].endswith("Anomaly"):
                exel["name"] = exel["AnomalyTestImage"]
                exel["x"] = exel["XTrackedTest"]
                exel["y"] = exel["YTrackedTest"]
                # Set respond value to "{SPACE}" for rows where TestStimulus.DEVICE is "Keyboard"
                exel.loc[exel['TestStimulus.DEVICE'] == 'Keyboard', 'respond'] = '{No_}'
                exel["task"] = "Anomaly"
                count_order += "3"
                print("File processed ", file)
            else:
                print("Skip file ", file)
                continue

            exel_list.append(exel)
        else:
            continue

    raw_data = pd.read_csv(asee_filename, dtype={5: str, 6: str})
    raw_data['Triggle Receive'] = pd.to_numeric(raw_data['Triggle Receive'], errors='coerce')

    raw_data.columns = raw_data.columns.str.strip()
    # Keep the new corresponding columns and the additional columns specified by the user
    raw_data = raw_data[
        ['Triggle Receive', 'Recording Time Stamp[ms]', 'Gaze Point X[px]', 'Gaze Point Y[px]', 'Fixation Index',
         'Fixation Duration[ms]', 'Fixation Point X[px]', 'Fixation Point Y[px]', 'Saccade Index',
         'Saccade Duration[ms]', 'Saccade Amplitude[px]', 'Saccade Velocity Average[px/ms]',
         'Saccade Single Velocity [px/ms]', 'Saccade Velocity Peak[px/ms]']]

    video_data = pd.DataFrame()

    for exel in exel_list:
        video_data = video_data._append(exel, ignore_index=True)

    # Simplify video_data to only keep columns "name", "x", "y", and "Respond"
    video_data = video_data[["name", "x", "y", "respond", "task"]]

    # Create new columns in the raw data to store the marker and video name values
    raw_data['marker'] = raw_data['Triggle Receive']
    raw_data['name'] = np.nan
    raw_data['x'] = np.nan
    raw_data['y'] = np.nan
    raw_data['respond'] = np.nan
    raw_data['task'] = np.nan

    # defining new dataframe for new process
    fixation_only = raw_data.copy(deep=True)
    removfix = raw_data.copy(deep=True)

    print("Matching markers...")

    # Initialize a counter for the video names
    video_counter = 0
    match = 0
    unmatch = []
    order_index = 0

    # Initialize variables to store start and end indices of samples and fitting rows
    sample_start_index, sample_end_index, fitting_start_index, fitting_end_index, sample_started, first_sample_processed = None, None, None, None, False, False

    print("Processing merging...")
    # Iterate over all rows in the raw data
    for index, row in raw_data.iterrows():
        # Check if this row has a start marker (7)
        if row['marker'] == 7:
            # print(f"Encountered start marker (7) at index {index}")
            # Set sample_start_index to current index if it is not already set
            if sample_start_index is None:
                sample_start_index = index
                removfix_start_index = sample_start_index

            # If sample_end_index is not set (i.e. we have not encountered an end marker), set it to the row before the current start row
            if sample_end_index is None:
                sample_end_index = index - 1

            # Process previous sample before starting new one if a sample has started
            if sample_started:
                order = (video_counter) // 105
                if last_order != order:
                    last_order = order
                    order_index = 0
                # print(f"Processing previous sample from index {sample_start_index} to {sample_end_index}")
                # Check if video_counter is less than number of rows in video_data
                if video_counter < len(video_data) and fitting_start_index is not None:
                    match += 1
                    print(
                        f"Matching fitting rows from index {fitting_start_index} to {fitting_end_index} with video_data at index {video_counter}")
                    # print(" video_data ", video_data.iloc[video_counter])
                    # Set name, x, y, and respond values for fitting rows in previous sample
                    digit_order = int(count_order[order])
                    time = raw_data.loc[fitting_end_index, 'Recording Time Stamp[ms]'] - \
                           raw_data.loc[fitting_start_index, 'Recording Time Stamp[ms]']
                    # print("current time ", time, "\n digit order ", digit_order, " \n index ", order_index)
                    #
                    img_name = os.path.basename(video_data.iloc[video_counter]['name'])
                    if img_name.startswith('itan'):
                        img_name = 't' + img_name
                    row_index = gt_list[gt_list['name'] == img_name].index
                    stat_data.iloc[row_index, digit_order] = time
                    stat_data.iloc[row_index, 4] = img_name
                    order_index += 1
                    raw_data.loc[fitting_start_index:fitting_end_index, 'name'] = video_data.iloc[video_counter]['name']
                    raw_data.loc[fitting_start_index:fitting_end_index, 'x'] = video_data.iloc[video_counter]['x']
                    raw_data.loc[fitting_start_index:fitting_end_index, 'y'] = video_data.iloc[video_counter]['y']
                    raw_data.loc[fitting_start_index:fitting_end_index, 'respond'] = video_data.iloc[video_counter][
                        'respond']
                    raw_data.loc[fitting_start_index:fitting_end_index, 'task'] = video_data.iloc[video_counter]['task']
                    # fixation only process
                    # Loop through rows from fitting_start_index to fitting_end_index
                    for i in range(fitting_start_index, fitting_end_index + 1):
                        # Check if this row does not have a value in either the "Fixation Index" or "Fixation Duration[ms]" column
                        if pd.isna(raw_data.at[i, 'Fixation Index']) and pd.isna(
                                raw_data.at[i, 'Fixation Duration[ms]']):
                            # Set the values of the 'Gaze Point X[px]' and 'Gaze Point Y[px]' columns to -1 for this row
                            fixation_only.at[i, 'Gaze Point X[px]'] = -1
                            fixation_only.at[i, 'Gaze Point Y[px]'] = -1
                    fixation_only.loc[fitting_start_index:fitting_end_index, 'name'] = video_data.iloc[video_counter]['name']
                    fixation_only.loc[fitting_start_index:fitting_end_index, 'x'] = video_data.iloc[video_counter]['x']
                    fixation_only.loc[fitting_start_index:fitting_end_index, 'y'] = video_data.iloc[video_counter]['y']
                    fixation_only.loc[fitting_start_index:fitting_end_index, 'respond'] = video_data.iloc[video_counter][
                        'respond']
                    fixation_only.loc[fitting_start_index:fitting_end_index, 'task'] = video_data.iloc[video_counter]['task']
                    # remove first fixation process
                    # Loop through rows from fitting_start_index to fitting_end_index
                    for i in range(fitting_start_index, fitting_end_index + 1):
                        # Check if this row has a value in either the "Saccade Index" or "Saccade Duration[ms]" column
                        if not pd.isna(raw_data.at[i, 'Saccade Index']) or not pd.isna(
                                raw_data.at[i, 'Saccade Duration[ms]']):
                            # Set fitting_start_index to the index of this row
                            removfix_start_index = i
                            break
                    removfix.loc[removfix_start_index:fitting_end_index, 'name'] = video_data.iloc[video_counter]['name']
                    removfix.loc[removfix_start_index:fitting_end_index, 'x'] = video_data.iloc[video_counter]['x']
                    removfix.loc[removfix_start_index:fitting_end_index, 'y'] = video_data.iloc[video_counter]['y']
                    removfix.loc[removfix_start_index:fitting_end_index, 'respond'] = video_data.iloc[video_counter][
                        'respond']
                    removfix.loc[removfix_start_index:fitting_end_index, 'task'] = video_data.iloc[video_counter][
                        'task']
                else:
                    unmatch.append(video_counter)
                    print(
                        f"Warning: Not enough rows in video_data to match all samples. Skipping sample from index {sample_start_index} to {sample_end_index}.")
                # Increment video counter
                video_counter += 1

                # Reset sample_start_index, sample_end_index, fitting_start_index, and fitting_end_index
                sample_start_index, sample_end_index, fitting_start_index, fitting_end_index = index, None, None, None
                removfix_start_index = sample_start_index

            # Set sample_started to True
            sample_started = True

        # Check if this row has a marker 3 or 4
        elif row['marker'] in [3, 4]:
            # print(f"Encountered marker {row['marker']} at index {index}")
            # Set fitting_start_index to current index if it is not already set
            if fitting_start_index is None:
                fitting_start_index = index
            # Set fitting_end_index to current index
            fitting_end_index = index

    print("Last Merge...")
    # Process last sample if it exists and a sample has started
    if sample_started and sample_start_index is not None:
        # If sample_end_index is not set (i.e. we have not encountered an end marker),
        # set it to the last row of the data
        if sample_end_index is None:
            sample_end_index = len(raw_data) - 1

        print(f"Processing last sample from index {sample_start_index} to {sample_end_index}")
        # Check if video_counter is less than number of rows in video_data
        if video_counter < len(video_data):
            print(
                f"Matching fitting rows from index {fitting_start_index} to {fitting_end_index} "
                f"with video_data at index {video_counter}")
            # Set name, x, y, and respond values for fitting rows in last sample
            raw_data.loc[fitting_start_index:fitting_end_index, 'name'] = video_data.iloc[video_counter]['name']
            raw_data.loc[fitting_start_index:fitting_end_index, 'x'] = video_data.iloc[video_counter]['x']
            raw_data.loc[fitting_start_index:fitting_end_index, 'y'] = video_data.iloc[video_counter]['y']
            raw_data.loc[fitting_start_index:fitting_end_index, 'respond'] = video_data.iloc[video_counter]['respond']
            raw_data.loc[fitting_start_index:fitting_end_index, 'task'] = video_data.iloc[video_counter]['task']
            time = raw_data.loc[fitting_end_index, 'Recording Time Stamp[ms]'] - \
                   raw_data.loc[fitting_start_index, 'Recording Time Stamp[ms]']
            img_name = os.path.basename(video_data.iloc[video_counter]['name'])
            if img_name.startswith('itan'):
                img_name = 't' + img_name
            row_index = gt_list[gt_list['name'] == img_name].index
            # print("current time ", time, "\n digit order ", digit_order, " \n index ", order_index)
            stat_data.iloc[row_index, digit_order] = time
            stat_data.iloc[row_index, 4] = img_name
            # fixation only process
            # Loop through rows from fitting_start_index to fitting_end_index
            for i in range(fitting_start_index, fitting_end_index + 1):
                # Check if this row does not have a value in either the "Fixation Index" or
                # "Fixation Duration[ms]" column
                if pd.isna(raw_data.at[i, 'Fixation Index']) and pd.isna(
                        raw_data.at[i, 'Fixation Duration[ms]']):
                    # Set the values of the 'Gaze Point X[px]' and 'Gaze Point Y[px]' columns to -1 for this row
                    fixation_only.at[i, 'Gaze Point X[px]'] = -1
                    fixation_only.at[i, 'Gaze Point Y[px]'] = -1
            fixation_only.loc[fitting_start_index:fitting_end_index, 'name'] = video_data.iloc[video_counter]['name']
            fixation_only.loc[fitting_start_index:fitting_end_index, 'x'] = video_data.iloc[video_counter]['x']
            fixation_only.loc[fitting_start_index:fitting_end_index, 'y'] = video_data.iloc[video_counter]['y']
            fixation_only.loc[fitting_start_index:fitting_end_index, 'respond'] = video_data.iloc[video_counter][
                'respond']
            fixation_only.loc[fitting_start_index:fitting_end_index, 'task'] = video_data.iloc[video_counter]['task']
            # remove first fixation process
            # Loop through rows from fitting_start_index to fitting_end_index
            for i in range(fitting_start_index, fitting_end_index + 1):
                # Check if this row has a value in either the "Saccade Index" or "Saccade Duration[ms]" column
                if not pd.isna(raw_data.at[i, 'Saccade Index']) or not pd.isna(
                        raw_data.at[i, 'Saccade Duration[ms]']):
                    # Set fitting_start_index to the index of this row
                    removfix_start_index = i
                    break
            removfix.loc[removfix_start_index:fitting_end_index, 'name'] = video_data.iloc[video_counter]['name']
            removfix.loc[removfix_start_index:fitting_end_index, 'x'] = video_data.iloc[video_counter]['x']
            removfix.loc[removfix_start_index:fitting_end_index, 'y'] = video_data.iloc[video_counter]['y']
            removfix.loc[removfix_start_index:fitting_end_index, 'respond'] = video_data.iloc[video_counter][
                'respond']
            removfix.loc[removfix_start_index:fitting_end_index, 'task'] = video_data.iloc[video_counter]['task']
        else:
            print(
                f"Warning: Not enough rows in video_data to match all samples. Skipping last sample from index "
                f"{sample_start_index} to {sample_end_index}.")
        # Increment video counter
        video_counter += 1

    print("matched number: ", match)
    print("unmatch length: ", len(unmatch), " \n unmatched list: ", unmatch)
    return raw_data, fixation_only, removfix, stat_data


# Check whether the output files exist
def output_exists(output_path, file):
    # 定义输出文件名
    marker_filename = os.path.join(output_path, file[:-4] + "_marker.csv")
    fixation_only_filename = os.path.join(output_path, file[:-4] + "_fixation_only.csv")
    removfix_filename = os.path.join(output_path, file[:-4] + "_removefix.csv")
    time_filename = os.path.join(output_path, file[:-4] + "_time.csv")

    # only return True if all output files exist
    return (os.path.exists(marker_filename) and
            os.path.exists(fixation_only_filename) and
            os.path.exists(removfix_filename) and
            os.path.exists(time_filename))


# Extract date and time from the filename
def extract_info(filename):
    # Express the filename using Regular Express
    match = re.search(r'raw_User(\d+)_(\d{2})(\d{2})(\d{2})(\d{6})_(\d{10}).csv', filename)
    if match:
        # 提取月份、日期和时间
        index = match.group(1)
        year = match.group(2)
        month = match.group(3)
        date = match.group(4)
        time = match.group(5)
        extract_time = match.group(6)
        # 返回提取到的信息
        return month, date, time
    else:
        # Return None if the filename does not match the expected format
        return None

if __name__ == '__main__':
    asee_path = "M:\\DRAMA_data\\actual\\raw_data"
    eprime_path = "M:\\DRAMA_data\\actual\\eprime_data"
    output_path = "M:\\DRAMA_data\\actual\\processed_data"

    # 获取asee_path中所有以.csv结尾的文件
    aseefiles = [file for file in os.listdir(asee_path) if file.endswith(".csv")]
    # 定义一个空列表，用于存储提取到的信息
    info_list = []

    # 遍历aseefiles中的每个文件
    for file in aseefiles:
        # 提取文件名中的信息
        info = extract_info(file)
        if info:
            # 如果提取到了信息，则将其添加到info_list中
            info_list.append((info, file))
    # 按照月份、日期和时间对info_list进行排序
    info_list.sort(key=lambda x: x[0])

    # 初始化索引计数器
    index_counter = 1

    # 初始化上一个月份和日期
    prev_month, prev_date = None, None

    # 遍历info_list中的每个元素
    for info, file in info_list:
        # 定义一个空列表，用于存储files
        files = []
        # 获取月份、日期和时间
        month, date, time = info
        print("Processing ", file)
        # 检查当前月份和日期是否与上一个月份和日期相同
        if month == prev_month and date == prev_date:
            # 如果相同，则递增索引计数器
            index_counter += 1
        else:
            # 如果不同，则重置索引计数器为1
            index_counter = 1

        # 更新上一个月份和日期
        prev_month, prev_date = month, date

        # 将索引计数器格式化为两位数字符串（例如01、02等）
        index_str = f"{index_counter:02d}"

        # 将月份、日期和索引拼接成字符串
        search_str = f"{month}{date}{index_str}"
        # 在eprime_path中查找包含search_str的xlsx文件
        for eprime_file in os.listdir(eprime_path):
            if eprime_file.endswith(".xlsx") and search_str in eprime_file:
                # 如果找到了符合条件的文件，则将其添加到files列表中
                files.append(eprime_file)
                # 将对应的asee csv文件名添加到asee_filename列表中
        asee_filename = os.path.join(asee_path, file)
        output_name = os.path.join(output_path, file)

        if output_exists(output_path, file):
            # 如果输出文件已经存在，则跳过处理过程
            print(f"Skipping {file} because output files already exist")
            continue

        if files == []:
            print("No matching files found for the asee file ", file)
            continue

        # 调用process_raw函数处理数据
        raw_data, fixation_only, removfix, stat_data = process_raw(asee_filename, eprime_path, files)
        # 保存更新后的数据到新的csv文件中
        marker_filename = output_name[:-4] + "_marker.csv"
        fixation_only_filename = output_name[:-4] + "_fixation_only.csv"
        removfix_filename = output_name[:-4] + "_removefix.csv"
        time_filename = output_name[:-4] + "_time.csv"
        stat_data.to_csv(time_filename, index=False)
        raw_data.to_csv(marker_filename, index=False)
        fixation_only.to_csv(fixation_only_filename, index=False)
        removfix.to_csv(removfix_filename, index=False)
        print("output to ", marker_filename)
    # # 显示结果
    # print("files:", files)
    # print("asee_filename:", asee_filename)
