import pandas as pd
import numpy as np

# Load the data from the three files
asee_filename = "collect_data_2023_7_25/DRAMA/DRAMA/User1_230725131316/User1_230725131316.csv"
eprime_filename = "Pilot-41901raw.xlsx"
# raw_data = pd.read_csv('0704\\ke_230704150443.csv')
raw_data = pd.read_csv(asee_filename)
raw_data.columns = raw_data.columns.str.strip()
triggle_name = asee_filename[:-4] + "_triggle2.csv"

trigger_data = pd.read_csv(triggle_name)
# video_data = pd.read_excel('0704\\Pilot-704-741.xlsx')
video_data = pd.read_excel(eprime_filename)
# Create new columns in the raw data to store the marker and video name values
raw_data['marker'] = np.nan
raw_data['video_name'] = np.nan

# Iterate over the rows in the trigger data
for index, row in trigger_data.iterrows():
    # Find the index of the first timestamp in the raw data that is larger than the current trigger timestamp
    match_index = raw_data[raw_data['timestamp'] >= row['timestamp']].index[0]
    # Set the marker value for this row in the raw data
    raw_data.at[match_index, 'marker'] = row['Value']

# # Filter the video data to only include rows where the second-to-last column is equal to 'ExpTrials'
# video_data = video_data[video_data.iloc[:, -2] == 'ExpTrials']

# Get the video names from the last column of the filtered video data
video_names = video_data.iloc[60:, -1]

# Define a list of all start markers
start_markers = [  # 101, 102, 103, 104, 105, 106, 107, 108,
                 201, 202, 203, 204, 205, 206, 207, 208]


# Initialize a counter for the video names
video_counter = 0

# Iterate over all rows in the raw data
for index, row in raw_data.iterrows():
    # Check if this row has a start marker
    if row['marker'] in start_markers:
        # # Set the video name for this row to be equal to current video name
        # raw_data.at[index, 'video_name'] = video_names.iloc[video_counter]
        # # Increment the video counter
        # video_counter += 1
        # Set the video name for this row to be equal to current video name
        raw_data.at[index, 'video_name'] = video_names.iloc[video_counter]
        # Increment the video counter
        video_counter += 1
        print(video_counter)

# Create a new DataFrame that only contains the desired columns
output_data = raw_data[['gazePoint.point.x', 'gazePoint.point.y', 'timestamp', 'marker', 'video_name']]

# Save the updated data to a new csv file
marker_filename = asee_filename[:-4] + "_marker.csv"
# output_data.to_csv('0704\\ke_230704150443_marker.csv', index=False)
output_data.to_csv(marker_filename, index=False)
print("output to ", marker_filename)

