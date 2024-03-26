import os
import pandas as pd
import re
import numpy as np
import json


def result_checker(turn_GT, hazard_GT, anomaly_GT, eprime_path, files):
    err_list_turn = []
    err_list_hazard = []
    err_list_anomaly = []
    exel_list = []
    count_order = ""
    # Search for any xlsx file starting with file_prefix and read them in the order of the numeric index in their
    files.sort(key=lambda x: int(x.split("-")[1]))
    for file in files:
        if file.endswith(".xlsx"):

            # initialise columns
            exel = pd.read_excel(os.path.join(eprime_path, file))
            exel["name"] = np.nan
            exel['x'] = np.nan
            exel['y'] = np.nan
            exel['respond'] = np.nan
            exel['task'] = np.nan

            # Drop the second to sixth row in the xlsx files
            exel = exel.drop(exel.index[[1, 2, 3, 4, 5]])

            # Sort the columns according to the task
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

    # initialise variables
    match_turn = 0
    unmatch_turn = 0
    match_hazard = 0
    unmatch_hazard = 0
    match_anomaly = 0
    unmatch_anomaly = 0
    true_Positives_turn = 0
    true_Negatives_turn = 0
    false_Positives_turn = 0
    false_Negatives_turn = 0

    for exel in exel_list:
        if exel["task"].iloc[1] == "Turn":
            # Compute F1 score for the Turn task
            for i in range(len(exel)):
                for j in range(len(turn_GT)):
                    if exel.iloc[i]["name"] == turn_GT.iloc[j]["name"]:
                        if exel.iloc[i]["respond"] == turn_GT.iloc[j]["respond"]:
                            match_turn += 1
                            if exel.iloc[i]["respond"] == "{ENTER}":
                                true_Positives_turn += 1
                            else:
                                true_Negatives_turn += 1
                        else:
                            err_list_turn.append(
                                [exel.iloc[i]["name"], exel.iloc[i]["respond"], turn_GT.iloc[j]["respond"]])
                            if exel.iloc[i]["respond"] == "{ENTER}":
                                false_Positives_turn += 1
                            else:
                                false_Negatives_turn += 1
                            print(exel.iloc[i]["name"], " ", exel.iloc[i]["respond"])
                            unmatch_turn += 1
            print(match_turn)
            print("unmatch turn", unmatch_turn)
            accuracy_turn = match_turn / (match_turn + unmatch_turn)
            if true_Positives_turn == 0 and false_Positives_turn == 0:
                false_Positives_turn += 1
            elif true_Positives_turn == 0 and false_Negatives_turn == 0:
                false_Negatives_turn += 1
            precision_turn = true_Positives_turn / (true_Positives_turn + false_Positives_turn)
            recall_turn = true_Positives_turn / (true_Positives_turn + false_Negatives_turn)

            if (precision_turn != 0) and (recall_turn != 0):
                f1_turn = 2 * (precision_turn * recall_turn) / (precision_turn + recall_turn)
            else:
                f1_turn = -1
        elif exel["task"].iloc[1] == "Hazard":
            # Compute accuracy for the Hazard task
            for i in range(len(exel)):
                hazard_name = exel.iloc[i]["name"]
                if isinstance(hazard_name, str):
                    hazard_name = hazard_name.split("\\")[-1]
                    if "itan" in hazard_name:
                        hazard_name = "t" + hazard_name
                    for j in range(len(hazard_GT)):
                        if hazard_name == hazard_GT.iloc[j]["name"]:
                            x = exel.iloc[i]["x"] - 1920
                            y = exel.iloc[i]["y"]
                            x1 = hazard_GT.iloc[j]["x1"] * 1920
                            y1 = hazard_GT.iloc[j]["y1"] * 1080
                            w = hazard_GT.iloc[j]["w"] * 1920
                            h = hazard_GT.iloc[j]["h"] * 1080
                            x2 = x1 + w
                            y2 = y1 + h
                            if x >= x1 and x <= x2 and y >= y1 and y <= y2:
                                match_hazard += 1
                            else:
                                unmatch_hazard += 1
                                err_list_hazard.append(exel.iloc[i]["name"])
            print("matched hazard ", match_hazard)
            print("unmatched hazard ", unmatch_hazard)
            accuracy_hazard = match_hazard / (match_hazard + unmatch_hazard)
        elif exel["task"].iloc[1] == "Anomaly":
            # Compute accuracy for the Anomaly task
            for i in range(len(exel)):
                anomaly_name = exel.iloc[i]["name"]
                if isinstance(anomaly_name, str):
                    anomaly_name = anomaly_name.split("\\")[-1]
                    if "itan" in anomaly_name:
                        anomaly_name = "t" + anomaly_name
                    for j in range(len(hazard_GT)):
                        if anomaly_name == anomaly_GT.iloc[j]["name"]:
                            x = exel.iloc[i]["x"] - 1920
                            y = exel.iloc[i]["y"]
                            x1 = anomaly_GT.iloc[j]["x1"] * 1920
                            y1 = anomaly_GT.iloc[j]["y1"] * 1080
                            x2 = anomaly_GT.iloc[j]["x2"] * 1920
                            y2 = anomaly_GT.iloc[j]["y2"] * 1080

                            if x1 <= x <= x2 and y1 <= y <= y2:
                                match_anomaly += 1
                            else:
                                unmatch_anomaly += 1
                                err_list_anomaly.append(exel.iloc[i]["name"])
            print("matched anomaly ", match_anomaly)
            print("unmatched anomaly ", unmatch_anomaly)
            accuracy_anomaly = match_anomaly / (match_anomaly + unmatch_anomaly)
    total_sample = match_hazard + unmatch_hazard + match_turn + unmatch_turn

    if total_sample != 60:
        # If samples number does not match
        print("ERROR!")
        print("total samples ", total_sample)
        print("turn match", match_turn, " unmatch", unmatch_turn)
        print("hazard match", match_hazard, " unmatch", unmatch_hazard)
    return err_list_turn, accuracy_turn, precision_turn, recall_turn, f1_turn, err_list_hazard, accuracy_hazard, \
        err_list_anomaly, accuracy_anomaly


# Extract time data from the filename
def extract_info(filename):
    # Use the Regular expression to distribute the data
    match = re.search(r'raw_User(\d+)_(\d{2})(\d{2})(\d{2})(\d{6})_(\d{10}).csv', filename)
    if match:
        # Extract time info
        index = match.group(1)
        year = match.group(2)
        month = match.group(3)
        date = match.group(4)
        time = match.group(5)
        extract_time = match.group(6)
        # Return the needed time data
        return month, date, time
    else:
        # Return None if the filename does not match the format
        return None


# Check whether the output files already exist
def output_exists(output_path, file):
    # Define the output filename
    marker_filename = os.path.join(output_path, file[:-4] + "_marker.csv")
    fixation_only_filename = os.path.join(output_path, file[:-4] + "_fixation_only.csv")
    removfix_filename = os.path.join(output_path, file[:-4] + "_removefix.csv")
    time_filename = os.path.join(output_path, file[:-4] + "_time.csv")

    # Check whether the output files already exist
    return (os.path.exists(marker_filename) and
            os.path.exists(fixation_only_filename) and
            os.path.exists(removfix_filename) and
            os.path.exists(time_filename))


if __name__ == '__main__':
    asee_path = "M:\\DRAMA_data\\actual\\raw_data"
    eprime_path = "M:\\DRAMA_data\\actual\\eprime_data"
    output_path = "M:\\DRAMA_data\\actual\\accuracy_data"
    turn_GT_path = "M:\\DRAMA_data\\actual\\TurnGroundTruth.xlsx"
    hazard_GT_path = "M:\\DRAMA_data\\actual\\drama_bbox_gt.json"
    anomaly_GT_path = "M:\\DRAMA_data\\actual\\anomaly_bbox_gt.json"

    turn_GT = pd.read_excel(turn_GT_path)

    with open(hazard_GT_path, 'r') as f:
        hazard_GT = json.load(f)

    # Convert the hazard_GT data into a DataFrame
    hazard_GT_df = pd.DataFrame(hazard_GT.items(), columns=['name', 'bbox'])
    hazard_GT_df[['x1', 'y1', 'w', 'h']] = pd.DataFrame(hazard_GT_df['bbox'].tolist(), index=hazard_GT_df.index)
    hazard_GT_df = hazard_GT_df.drop('bbox', axis=1)

    # Load the anomaly_GT JSON file
    with open("M:\\DRAMA_data\\actual\\anomaly_bbox_gt.json", 'r') as f:
        anomaly_GT = json.load(f)
    # Convert the anomaly_GT data into a DataFrame
    anomaly_data = []
    for name, values in anomaly_GT.items():
        # Get the bounding box information for the current object
        bbox = values['bbox']
        # Unpack the bounding box values into separate variables
        x1, y1, x2, y2 = bbox
        complete_name = name + ".jpeg"
        # Append a new row to the gt_data list
        anomaly_data.append([complete_name, x1, y1, x2, y2])

    anomaly_GT_df = pd.DataFrame(anomaly_data, columns=['name', 'x1', 'y1', 'x2', 'y2'])


    # Read all csv files under asee_path
    aseefiles = [file for file in os.listdir(asee_path) if file.endswith(".csv")]
    # initialise array for later storage
    info_list = []
    acc_list_turn = []
    acc_list_hazard = []
    acc_list_anomaly = []

    for file in aseefiles:
        # Extract the files time data
        info = extract_info(file)
        if info:
            # Record the file if it has the correct format
            info_list.append((info, file))
    # Rearrange the files in increasing order
    info_list.sort(key=lambda x: x[0])

    # initialise variables
    index_counter = 1
    prev_month, prev_date = None, None

    for info, file in info_list:
        # initialise the storage list
        files = []
        month, date, time = info
        print("Processing ", file)
        # Check if the current record belongs to the same date of the last record, and count the index
        if month == prev_month and date == prev_date:
            index_counter += 1
        else:
            index_counter = 1

        # Update the info of last record
        prev_month, prev_date = month, date

        # Turn the index into two digits string
        index_str = f"{index_counter:02d}"

        # Reformat the time data
        search_str = f"{month}{date}{index_str}"
        # Search for the corresponding file according to the time data
        for eprime_file in os.listdir(eprime_path):
            if eprime_file.endswith(".xlsx") and search_str in eprime_file:
                files.append(eprime_file)

        output_name = os.path.join(output_path, file)

        if output_exists(output_path, file):
            # if output file exist, then skip
            print(f"Skipping {file} because output files already exist")
            continue

        if files == []:
            print("No matching files found for the asee file ", file)
            continue

        # Save the computed infomation
        errlist_turn, acc_result_turn, prcision_turn, recall_turn, f1_turn, errlist_hazard, acc_hazard, \
            errlist_anomaly, acc_anomaly = \
            result_checker(turn_GT, hazard_GT_df, anomaly_GT_df, eprime_path, files)
        acc_list_turn.append([search_str, acc_result_turn, prcision_turn, recall_turn, f1_turn])
        acc_filename_turn = output_path + "\\" + search_str + "_acc_" + str(acc_result_turn)[:4] + "_turn.csv"
        acc_df_turn = pd.DataFrame([acc_list_turn[-1]], columns=["name", "acc", "precision", "recall", "f1"])
        acc_df_turn.to_csv(acc_filename_turn, index=False)

        sum_filename_turn = output_path + "\\sum_data_turn.csv"
        sum_df_turn = pd.DataFrame(acc_list_turn, columns=["name", "acc", "precision", "recall", "f1"])
        sum_df_turn.to_csv(sum_filename_turn)

        error_df_turn = pd.DataFrame(errlist_turn, columns=["name", "respond", "GroundTruth"])
        error_df_turn.to_csv(output_path + "\\" + search_str + "_err_turn.csv")

        acc_list_hazard.append([search_str, acc_hazard])
        acc_filename_hazard = output_path + "\\" + search_str + "_acc_" + str(acc_hazard)[:4] + "_hazard.csv"
        acc_df_hazard = pd.DataFrame([acc_list_hazard[-1]], columns=["name", "acc"])
        acc_df_hazard.to_csv(acc_filename_hazard, index=False)

        sum_filename_hazard = output_path + "\\sum_data_hazard.csv"
        sum_df_hazard = pd.DataFrame(acc_list_hazard, columns=["name", "acc"])
        sum_df_hazard.to_csv(sum_filename_hazard)

        error_df_hazard = pd.DataFrame(errlist_hazard, columns=["name"])
        error_df_hazard.to_csv(output_path + "\\" + search_str + "_err_hazard.csv")

        acc_list_anomaly.append([search_str, acc_anomaly])
        acc_filename_anomaly = output_path + "\\" + search_str + "_acc_" + str(acc_anomaly)[:4] + "_anomaly.csv"
        acc_df_anomaly = pd.DataFrame([acc_list_anomaly[-1]], columns=["name", "acc"])
        acc_df_anomaly.to_csv(acc_filename_anomaly, index=False)

        sum_filename_anomaly = output_path + "\\sum_data_anomaly.csv"
        sum_df_anomaly = pd.DataFrame(acc_list_anomaly, columns=["name", "acc"])
        sum_df_anomaly.to_csv(sum_filename_anomaly)

        error_df_anomaly = pd.DataFrame(errlist_anomaly, columns=["name"])
        error_df_anomaly.to_csv(output_path + "\\" + search_str + "_err_anomaly.csv")
