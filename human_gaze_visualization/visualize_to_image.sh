DATE="10_30"
GAUSSIAN=55
DATA_PATH="all_new_old_driver_v3"
CHOOSE=0

echo $DATE
echo $GAUSSIAN


# get single and mean
python gen_gaze_from_ysh_processed_2_rm_cb_single.py --root_time $DATE --gaussian $GAUSSIAN --split "new" --choose_driver $CHOOSE --data_path $DATA_PATH &
python gen_gaze_from_ysh_processed_2_rm_cb_single.py --root_time $DATE --gaussian $GAUSSIAN --split "old" --choose_driver $CHOOSE --data_path $DATA_PATH &


python gen_gaze_from_ysh_processed_2_rm_cb_mean.py --root_time $DATE --gaussian $GAUSSIAN --split "new" --choose_driver $CHOOSE --data_path $DATA_PATH &
python gen_gaze_from_ysh_processed_2_rm_cb_mean.py --root_time $DATE --gaussian $GAUSSIAN --split "old" --choose_driver $CHOOSE --data_path $DATA_PATH &

wait
echo "single and mean done"

python convert_mean_name.py --root_time $DATE --gaussian $GAUSSIAN --split "new" &
python convert_mean_name.py --root_time $DATE --gaussian $GAUSSIAN --split "old" &
wait
echo "convert mean name"


python concat_all_times.py --root_time $DATE --gaussian $GAUSSIAN --split "new" --choose_driver $CHOOSE --data_path $DATA_PATH &
python concat_all_times.py --root_time $DATE --gaussian $GAUSSIAN --split "old" --choose_driver $CHOOSE --data_path $DATA_PATH &

python concat_all_times_for_mean.py --root_time $DATE --gaussian $GAUSSIAN --split "new" &
python concat_all_times_for_mean.py --root_time $DATE --gaussian $GAUSSIAN --split "old" &

wait
echo "concat all times done"


python concat_drivers.py --root_time $DATE --gaussian $GAUSSIAN --split "new" --choose_driver 1 --data_path $DATA_PATH &
python concat_drivers.py --root_time $DATE --gaussian $GAUSSIAN --split "old" --choose_driver 1 --data_path $DATA_PATH &


wait
echo "concat all new drivers done"
echo "concat all old drivers done"

python concat_new_old_drivers_for_mean.py --root_time $DATE --gaussian $GAUSSIAN &

python concat_new_old_drivers.py --root_time $DATE  &

wait
echo "concat new old drivers done"

python concat_image_new_old_mean.py --root_time $DATE --gaussian $GAUSSIAN &

wait
echo "all done"

echo $DATE
echo $GAUSSIAN