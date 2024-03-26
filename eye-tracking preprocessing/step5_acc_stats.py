import pandas as pd
import os

path = "M:\\DRAMA_data\\actual\\accuracy_data"

# Hazard condition
hazard_files = []
for subdir, dirs, files in os.walk(path):
    for file in files:
        if "sum_data_hazard" in file and file.endswith(".csv"):
            hazard_files.append(os.path.join(subdir, file))

# Hazard condition
hazard_data = pd.DataFrame(columns=["name", "acc", "condition"])
for f in hazard_files:
    # Read the file
    data = pd.read_csv(f, usecols=["name", "acc"])
    # Extract the condition from the filename
    condition = f.split("\\")[4]
    # Add the condition column
    data["condition"] = condition
    # Append to the hazard_data DataFrame
    hazard_data = hazard_data._append(data)

# Group by condition and calculate statistics
hazard_data_grouped = hazard_data.groupby(["condition"])
for name, group in hazard_data_grouped:
    print(f"Condition: {name}")
    print(f"Average accuracy: {group['acc'].mean()}")
    print(f"Standard deviation of accuracy: {group['acc'].std()}")

# Anomaly condition
anomaly_files = []
for subdir, dirs, files in os.walk(path):
    for file in files:
        if "sum_data_anomaly" in file and file.endswith(".csv"):
            anomaly_files.append(os.path.join(subdir, file))

# Anomaly condition
anomaly_data = pd.DataFrame(columns=["name", "acc", "condition"])
for f in anomaly_files:
    # Read the file
    data = pd.read_csv(f, usecols=["name", "acc"])
    # Extract the condition from the filename
    condition = f.split("\\")[4]
    # Add the condition column
    data["condition"] = condition
    # Append to the anomaly_data DataFrame
    anomaly_data = anomaly_data._append(data)

# Group by condition and calculate statistics
anomaly_data_grouped = anomaly_data.groupby(["condition"])
for name, group in anomaly_data_grouped:
    print(f"Condition: {name}")
    print(f"Average accuracy: {group['acc'].mean()}")
    print(f"Standard deviation of accuracy: {group['acc'].std()}")

# Turn condition
turn_files = []
for subdir, dirs, files in os.walk(path):
    for file in files:
        if "sum_data_turn" in file and file.endswith(".csv"):
            turn_files.append(os.path.join(subdir, file))

# Turn condition
turn_data = pd.DataFrame(columns=["name", "acc", "precision", "recall", "f1", "condition"])
for f in turn_files:
    # Read the file
    data = pd.read_csv(f, usecols=["name", "acc", "precision", "recall", "f1"])
    # Extract the condition from the filename
    condition = f.split("\\")[4]
    # Add the condition column
    data["condition"] = condition
    # Append to the turn_data DataFrame
    turn_data = turn_data._append(data)

# Group by condition and calculate statistics
turn_data_grouped = turn_data.groupby(["condition"])
for name, group in turn_data_grouped:
    print(f"Condition: {name}")
    print(f"Average accuracy: {group['acc'].mean()}")
    print(f"Standard deviation of accuracy: {group['acc'].std()}")
    print(f"Average precision: {group['precision'].mean()}")
    print(f"Standard deviation of precision: {group['precision'].std()}")
    print(f"Average recall: {group['recall'].mean()}")
    print(f"Standard deviation of recall: {group['recall'].std()}")
    print(f"Average F1 score: {group['f1'].mean()}")
    print(f"Standard deviation of F1 score: {group['f1'].std()}")

# Save to a new csv file
result_df = pd.DataFrame(columns=["condition", "accuracy_mean", "accuracy_std", "precision_mean", "precision_std",
                                  "recall_mean", "recall_std", "f1_mean", "f1_std"])
for name, group in hazard_data_grouped:
    result_df = result_df._append({"condition": name + "_hazard",
                                  "accuracy_mean": group["acc"].mean(),
                                  "accuracy_std": group["acc"].std()}, ignore_index=True)
result_df = result_df._append({"condition": "all_hazard",
                                  "accuracy_mean": hazard_data["acc"].mean(),
                                  "accuracy_std": hazard_data["acc"].std()}, ignore_index=True)

for name, group in anomaly_data_grouped:
    result_df = result_df._append({"condition": name + "_anomaly",
                                   "accuracy_mean": group["acc"].mean(),
                                   "accuracy_std": group["acc"].std()}, ignore_index=True)
result_df = result_df._append({"condition": "all_anomaly",
                               "accuracy_mean": anomaly_data["acc"].mean(),
                               "accuracy_std": anomaly_data["acc"].std()}, ignore_index=True)

for name, group in turn_data_grouped:
    result_df = result_df._append({"condition": name + "_turn",
                                  "accuracy_mean": group["acc"].mean(),
                                  "accuracy_std": group["acc"].std(),
                                  "precision_mean": group["precision"].mean(),
                                  "precision_std": group["precision"].std(),
                                  "recall_mean": group["recall"].mean(),
                                  "recall_std": group["recall"].std(),
                                  "f1_mean": group["f1"].mean(),
                                  "f1_std": group["f1"].std()}, ignore_index=True)
result_df = result_df._append({"condition": "all_turn",
                                  "accuracy_mean": turn_data["acc"].mean(),
                                  "accuracy_std": turn_data["acc"].std(),
                                  "precision_mean": turn_data["precision"].mean(),
                                  "precision_std": turn_data["precision"].std(),
                                  "recall_mean": turn_data["recall"].mean(),
                                  "recall_std": turn_data["recall"].std(),
                                  "f1_mean": turn_data["f1"].mean(),
                                  "f1_std": turn_data["f1"].std()}, ignore_index=True)

result_df.to_csv(os.path.join(path, 'result.csv'), index=False)