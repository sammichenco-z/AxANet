DATA_PATH="all_new_old_driver_final/final_version_2/compute_dataset/t0_t1"
CHOOSE=False

python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t0_t1_65_oia" --gaussian 65 --split "new" --data_path $DATA_PATH &
python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t0_t1_65_oia" --gaussian 65 --split "old" --data_path $DATA_PATH &

python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t0_t1_35_hazard_anomaly" --gaussian 35 --split "new" --data_path $DATA_PATH &
python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t0_t1_35_hazard_anomaly" --gaussian 35 --split "old" --data_path $DATA_PATH &

wait
echo "t0_t1 done"

python pad_zero_numpy.py --root_time "final_version_compute_dataset_t0_t1_65_oia" &
python pad_zero_numpy.py --root_time "final_version_compute_dataset_t0_t1_35_hazard_anomaly" &

wait
echo "t0_t1 done"

DATA_PATH="all_new_old_driver_final/final_version_2/compute_dataset/t1_t2"
CHOOSE=False

python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t1_t2_65_oia" --gaussian 65 --split "new" --data_path $DATA_PATH &
python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t1_t2_65_oia" --gaussian 65 --split "old" --data_path $DATA_PATH &

python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t1_t2_35_hazard_anomaly" --gaussian 35 --split "new" --data_path $DATA_PATH &
python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t1_t2_35_hazard_anomaly" --gaussian 35 --split "old" --data_path $DATA_PATH &

wait
echo "t1_t2 done"

python pad_zero_numpy.py --root_time "final_version_compute_dataset_t1_t2_65_oia" &
python pad_zero_numpy.py --root_time "final_version_compute_dataset_t1_t2_35_hazard_anomaly" &

wait
echo "t1_t2 done"

DATA_PATH="all_new_old_driver_final/final_version_2/compute_dataset/t2_end"
CHOOSE=False

python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t2_end_65_oia" --gaussian 65 --split "new" --data_path $DATA_PATH &
python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t2_end_65_oia" --gaussian 65 --split "old" --data_path $DATA_PATH &

python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t2_end_35_hazard_anomaly" --gaussian 35 --split "new" --data_path $DATA_PATH &
python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t2_end_35_hazard_anomaly" --gaussian 35 --split "old" --data_path $DATA_PATH &

wait
echo "t2_end done"

python pad_zero_numpy.py --root_time "final_version_compute_dataset_t2_end_65_oia" &
python pad_zero_numpy.py --root_time "final_version_compute_dataset_t2_end_35_hazard_anomaly" &

wait
echo "t2_end done"


# DATA_PATH="all_new_old_driver_final/final_version_2/compute_dataset/t0_end"
# CHOOSE=False

# python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t0_end_65_oia" --gaussian 65 --split "new" --data_path $DATA_PATH &
# python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t0_end_65_oia" --gaussian 65 --split "old" --data_path $DATA_PATH &

# python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t0_end_35_hazard_anomaly" --gaussian 35 --split "new" --data_path $DATA_PATH &
# python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t0_end_35_hazard_anomaly" --gaussian 35 --split "old" --data_path $DATA_PATH &

# wait
# echo "t0_end done"

# python pad_zero_numpy.py --root_time "final_version_compute_dataset_t0_end_65_oia" &
# python pad_zero_numpy.py --root_time "final_version_compute_dataset_t0_end_35_hazard_anomaly" &

# wait
# echo "t0_end done"


# DATA_PATH="all_new_old_driver_final/final_version_2/compute_dataset/t0_end_all"
# CHOOSE=False

# python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t0_end_all_65_oia" --gaussian 65 --split "new" --data_path $DATA_PATH &
# python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t0_end_all_65_oia" --gaussian 65 --split "old" --data_path $DATA_PATH &

# python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t0_end_all_35_hazard_anomaly" --gaussian 35 --split "new" --data_path $DATA_PATH &
# python gen_gaze_from_ysh_processed_2_rm_cb_mean_npy.py --root_time "final_version_compute_dataset_t0_end_all_35_hazard_anomaly" --gaussian 35 --split "old" --data_path $DATA_PATH &

# wait
# echo "t0_end_all done"

# python pad_zero_numpy.py --root_time "final_version_compute_dataset_t0_end_all_65_oia" &
# python pad_zero_numpy.py --root_time "final_version_compute_dataset_t0_end_all_35_hazard_anomaly" &

# wait
# echo "t0_end_all done"
