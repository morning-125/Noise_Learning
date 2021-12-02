gpu_device=0
test_path=/data1/qd/noise_master/ag_news/ag_news_csv/test.csv

task=base
noise_type=asym
noise_ratio=0.3

model_dir=/data1/qd/noise_master/save_model/$task/$noise_ratio$noise_type
model_path=${model_dir}_last.ckpt

CUDA_VISIBLE_DEVICES=$gpu_device python /home/qd/code/noise_master/test.py \
    --config  /home/qd/code/noise_master/config/default.config \
    --test_path $test_path \
    --model_path $model_path \
    --show_bar 
