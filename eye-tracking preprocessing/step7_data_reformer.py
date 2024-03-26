import pandas as pd
import numpy as np
import glob
import os

# Define the paths
# paths = ['M:\\DRAMA_data\\actual\\re-examine_script\\new\\*', 'M:\\DRAMA_data\\actual\\gaze_in_box\\personalised_aoi\\*']
paths = ['M:\\DRAMA_data\\actual\\re-examine_script\\*'] # , 'M:\\DRAMA_data\\actual\\gaze_in_box\\personalised_aoi\\*'
# Define the output directory
output_dir = 'M:\\DRAMA_data\\actual\\re-examine_script\\reformat'

# Check if output directory exists, if not create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop over the paths
for path in paths:
    # Find all CSV files that contain "first" in their filename
    for filename in glob.glob(path):
        print("PROCESSING ", filename)
        # Load the data
        df = pd.read_csv(filename)
        df_1 = pd.read_csv(filename)
        df_nan = pd.read_csv(filename)

        # Remove -1 condition
        df = df.replace(-1, np.nan)

        # # Replace -1 and NaN with 0
        # df = df.replace(-1, 0)
        # df = df.fillna(0)
        if "fix_avg" in df:
            df = df.replace(0, np.nan)

        # Define the conditions for the sets
        conditions = [
            (df['img_id'] >= 0) & (df['img_id'] <= 29),
            (df['img_id'] >= 30) & (df['img_id'] <= 59),
            (df['img_id'] >= 60) & (df['img_id'] <= 89)
        ]

        # Define the set names
        values = ['DRAMA', 'OIA', 'ANOMALY']

        # Create a new column with the set names
        df['set'] = np.select(conditions, values)
        df_1['set'] = np.select(conditions, values)
        df_nan['set'] = np.select(conditions, values)

        # Initialize an empty DataFrame to store the final results
        final_df = pd.DataFrame()
        final_nan_df = pd.DataFrame()
        final_zero_df = pd.DataFrame()

        # For each task, calculate the mean, ignoring 0 for files without 'unique' in the filename
        for task in ['hazard_task', 'turn_task', 'anomaly_task']:
            # if 'unique' in filename:
            #     df[task + '_mean'] = df[task]
            # else:
            #     df[task + '_mean'] = df[task].apply(lambda x: np.nan if x == 0 else x)
            df[task + '_mean'] = df[task]
            nan_count = df_nan.groupby(['driver_id', 'set'])[task].apply(lambda x: x.isna().sum()).reset_index()
            zero_count = df_1.groupby(['driver_id', 'set'])[task].apply(lambda x: (x == -1).sum()).reset_index()

            nan_count.fillna(0, inplace=True)
            zero_count.fillna(0, inplace=True)

            # df[task + '_mean'] = df[task].apply(lambda x: np.nan if x == 0 else x)

            grouped = df.groupby(['driver_id', 'set'])[task + '_mean'].mean()

            # Pivot the DataFrame to get the desired format and rename the columns as per your requirements
            pivot_df = grouped.reset_index().pivot(index='driver_id', columns='set', values=task + '_mean')
            pivot_nan_df = nan_count.pivot(index='driver_id', columns='set', values=task)
            pivot_zero_df = zero_count.pivot(index='driver_id', columns='set', values=task)

            pivot_df.columns = [f'{col}_{task.split("_")[0]}' for col in pivot_df.columns]
            pivot_nan_df.columns = [f'{col}_{task.split("_")[0]}' for col in pivot_nan_df.columns]
            pivot_zero_df.columns = [f'{col}_{task.split("_")[0]}' for col in pivot_zero_df.columns]

            pivot_df.columns = [f'{col.split("_")[0].upper()}_{col.split("_")[1].lower()}' for col in
                                        pivot_df.columns]
            pivot_nan_df.columns = [f'{col.split("_")[0].upper()}_{col.split("_")[1].lower()}' for col in
                                    pivot_nan_df.columns]
            pivot_zero_df.columns = [f'{col.split("_")[0].upper()}_{col.split("_")[1].lower()}' for col in
                                     pivot_zero_df.columns]
            # Reset the index and rename the index column to 'Expertise'
            pivot_df.reset_index(inplace=True)
            pivot_nan_df.reset_index(inplace=True)
            pivot_zero_df.reset_index(inplace=True)
            print()

            # Merge this DataFrame with the final DataFrame
            if final_df.empty:
                final_df = pivot_df.copy()
            else:
                final_df = pd.merge(final_df, pivot_df, left_index=True, right_index=True)

            if final_zero_df.empty:
                final_zero_df = pivot_zero_df.copy()
            else:
                final_zero_df = pd.merge(final_zero_df, pivot_zero_df, left_index=True, right_index=True)

            if final_nan_df.empty:
                final_nan_df = pivot_nan_df.copy()
            else:
                final_nan_df = pd.merge(final_nan_df, pivot_nan_df, left_index=True, right_index=True)
        # Reset the index and rename the index column to 'Expertise'
        final_df.reset_index(inplace=True)
        final_zero_df.reset_index(inplace=True)
        final_nan_df.reset_index(inplace=True)
        # print(final_nan_df)

        # Add these lines to map 'driver_id' to 'Expertise'
        conditions = [
            (final_df['driver_id'] >= 1) & (final_df['driver_id'] <= 18),
            (final_df['driver_id'] >= 19) & (final_df['driver_id'] <= 36)
        ]
        values = ['novice', 'expert']
        final_df['Expertise'] = np.select(conditions, values)

        nan_conditions = [
            (final_nan_df['driver_id'] >= 1) & (final_nan_df['driver_id'] <= 18),
            (final_nan_df['driver_id'] >= 19) & (final_nan_df['driver_id'] <= 36)
        ]
        final_nan_df['Expertise'] = np.select(nan_conditions, values)

        zero_conditions = [
            (final_zero_df['driver_id'] >= 1) & (final_zero_df['driver_id'] <= 18),
            (final_zero_df['driver_id'] >= 19) & (final_zero_df['driver_id'] <= 36)
        ]
        final_zero_df['Expertise'] = np.select(zero_conditions, values)

        # Reorder the columns as per your requirements
        final_df = final_df[
            ['Expertise', 'DRAMA_hazard', 'DRAMA_turn', 'DRAMA_anomaly', 'OIA_hazard', 'OIA_turn', 'OIA_anomaly',
             'ANOMALY_hazard', 'ANOMALY_turn', 'ANOMALY_anomaly']]
        final_nan_df = final_nan_df[
            ['Expertise', 'DRAMA_hazard', 'DRAMA_turn', 'DRAMA_anomaly', 'OIA_hazard', 'OIA_turn',
             'OIA_anomaly', 'ANOMALY_hazard', 'ANOMALY_turn', 'ANOMALY_anomaly']]
        final_zero_df = final_zero_df[
            ['Expertise', 'DRAMA_hazard', 'DRAMA_turn', 'DRAMA_anomaly', 'OIA_hazard', 'OIA_turn',
             'OIA_anomaly', 'ANOMALY_hazard', 'ANOMALY_turn', 'ANOMALY_anomaly']]

        # Save to a new CSV file with a name based on the original filename in the specified output directory
        base_filename = os.path.basename(filename).replace('first_before_aoi', '')
        nan_output_filename = os.path.basename(filename).replace('fixationavg', 'nan')
        zero_output_filename = os.path.basename(filename).replace('fixationavg', 'minusone')

        output_filename = os.path.join(output_dir, f'SPSS_{base_filename}')
        nan_output_filename = os.path.join(output_dir, f'SPSS_{nan_output_filename}')
        zero_output_filename = os.path.join(output_dir, f'SPSS_{zero_output_filename}')

        final_df.to_csv(output_filename, index=False)
        final_nan_df.to_csv(nan_output_filename, index=False)
        final_zero_df.to_csv(zero_output_filename, index=False)
