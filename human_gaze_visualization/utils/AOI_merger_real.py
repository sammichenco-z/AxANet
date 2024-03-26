import os
import pandas as pd

# Load the marker data
asee_filename = "User1_230419192801.csv"
# marker_data = pd.read_csv('0704\\ke_230704150443_marker.csv')
marker_filename = asee_filename[:-4] + "_marker.csv"
marker_data = pd.read_csv(marker_filename)

# Define the start and end markers
start_markers = [# 101, 102, 103, 104, 105, 106, 107, 108,
                 201, 202, 203, 204, 205, 206, 207, 208]

end_markers = [# 111, 112, 113, 114, 115, 116, 117, 118,
               211, 212, 213, 214, 215, 216, 217, 218]
matched = 0

# Define the label map as a list of tuples
label_map = [
    ('DD-CR', 'DD'),
    ('DD-HR', 'DD'),
    ('DD-OR', 'DD'),
    ('DD-OHR', 'DD'),
    ('Turn-CR', 'Turn'),
    ('Vehicle-HR', 'Vehicle'),
    ('Vehicle-OHR', 'Vehicle'),
    ('Occlusion-OR', 'Occlusion'),
    ('Occlusion-OHR', 'Occlusion')
]

# Define the frame duration
frame_duration = 33000

# Create a new DataFrame to store max_id, actual_frames_matched and corresponding video_name values
new_data = pd.DataFrame(columns=['max_id', 'actual_frames_matched', 'video_name'])

# Initialize the last frame number variable
last_frame_number = None

# Iterate over the rows in the marker data
for index, row in marker_data.iterrows():
    # Check if this row has a start marker
    if row['marker'] in start_markers:
        # Get the video name for this row
        video_name = row['video_name']
        # Normalize the video name by replacing all forward slashes with backward slashes and removing any leading or trailing whitespace
        video_name = video_name.replace('/', '\\').strip()
        # Check if the video name starts with "experiment\\real\\"
        # if video_name.lower().startswith('experiment\\real\\'):
        if video_name.lower().startswith('main\\real\\'): # main for old experiment record
            # Get the filename of the csv file by extracting the last part of the video name after the last backslash
            csv_filename = video_name.split('\\')[-1]
            # Remove the file extension from the csv filename
            csv_filename = os.path.splitext(csv_filename)[0]

            # Normalize the csv filename by removing any leading or trailing whitespace and converting it to lowercase
            csv_filename = csv_filename.strip().lower()
            # Construct the path to the csv file
            csv_path = os.path.join('CVAT AOI', csv_filename, csv_filename + '.csv')
            # Check if the csv file exists
            print("video found ", csv_filename, " the path is ", csv_path)
            if os.path.exists(csv_path):
                matched += 1
                # Load the subfolder csv data
                subfolder_csv_data = pd.read_csv(csv_path)
                # Get the base timestamp for this start marker
                base_timestamp = row['timestamp']
                # Find the index of the next end marker
                end_index = marker_data[(marker_data['marker'].isin(end_markers)) & (marker_data.index > index)].index[0]
                # Iterate over all rows between this start marker and the next end marker
                for i in range(index, end_index + 1):
                    # Get the timestamp for this row
                    timestamp = marker_data.at[i, 'timestamp']
                    # Calculate the frame number based on the timestamp and base timestamp
                    frame_number = (timestamp - base_timestamp) // frame_duration
                    # Check if there are rows in the subfolder csv data with this frame number as their id
                    subfolder_csv_rows = subfolder_csv_data[subfolder_csv_data['id'] == frame_number]
                    if not subfolder_csv_rows.empty:
                        # Get the name for this frame from the first row in the subfolder csv data
                        name = subfolder_csv_rows.iloc[0]['name']
                        # Set the video name for this row to be equal to the name from the subfolder csv data
                        marker_data.at[i, 'video_name'] = name

                        # Get a list of all unique labels in the subfolder csv rows for this frame number
                        unique_labels = subfolder_csv_rows['label'].unique()
                        # Iterate over all unique labels
                        for label in unique_labels:
                            # Check if there are rows in the subfolder csv data with this label
                            label_rows = subfolder_csv_rows[subfolder_csv_rows['label'] == label]
                            if not label_rows.empty:
                                # Get the xtl, ytl, xbr, and ybr values for this label from the first row in the subfolder csv data
                                xtl = label_rows.iloc[0]['xtl']
                                ytl = label_rows.iloc[0]['ytl']
                                xbr = label_rows.iloc[0]['xbr']
                                ybr = label_rows.iloc[0]['ybr']

                                # Get the original label by splitting the label at the first space character
                                original_label = label.split(' ', 1)[0]
                                # Check if the original label is in the label map
                                if any(original_label == original for original, merged in label_map):
                                    # Get the merged label from the label map
                                    merged_label = next(merged for original, merged in label_map if original_label == original)
                                    # Set the xtl, ytl, xbr, and ybr values for this row and merged label in the marker data
                                    marker_data.at[i, f'{merged_label} xtl'] = xtl
                                    marker_data.at[i, f'{merged_label} ytl'] = ytl
                                    marker_data.at[i, f'{merged_label} xbr'] = xbr
                                    marker_data.at[i, f'{merged_label} ybr'] = ybr
                                else:
                                    # Set the xtl, ytl, xbr, and ybr values for this row and label in the marker data
                                    marker_data.at[i, f'{label} xtl'] = xtl
                                    marker_data.at[i, f'{label} ytl'] = ytl
                                    marker_data.at[i, f'{label} xbr'] = xbr
                                    marker_data.at[i, f'{label} ybr'] = ybr

                # Record maximum id in video name csv and actual number of frames being matched at last marker line
                max_id = subfolder_csv_data['id'].max()
                actual_frames_matched = int(name[-6:-4])
                new_row = pd.DataFrame({'max_id': [max_id],
                                        'actual_frames_matched': [actual_frames_matched],
                                        'video_name': [video_name]})
                new_data = pd.concat([new_data, new_row], ignore_index=True)
print(matched)
# Save new data to a new csv file
new_data.to_csv('0704\\max_id_actual_frames_matched.csv', index=False)

# Define a list of columns to keep
columns_to_keep = ['gazePoint.point.x', 'gazePoint.point.y', 'timestamp', 'marker', 'video_name', 'DD xtl', 'DD ytl', 'DD xbr', 'DD ybr', 'Turn xtl', 'Turn ytl', 'Turn xbr', 'Turn ybr', 'Vehicle xtl', 'Vehicle ytl', 'Vehicle xbr', 'Vehicle ybr', 'Occlusion xtl', 'Occlusion ytl', 'Occlusion xbr', 'Occlusion ybr']

# Keep only the columns in the columns_to_keep list
marker_data = marker_data[columns_to_keep]

merged_r_filename = asee_filename[:-4] + "_Rmerged.csv"

# Save updated marker data to a new csv file without changing current save setting to file
# marker_data.to_csv('0704\\ke_230704150443_marker_updated.csv', index=False)
marker_data.to_csv(merged_r_filename, index=False)
print("output to ", merged_r_filename)
