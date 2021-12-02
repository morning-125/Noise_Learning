gpu_device=6

train_path=/data1/qd/noise_master/ag_news/ag_news_csv/train.csv

task=base
noise_type=asym
noise_ratio=0.5

model_dir=/data1/qd/noise_master/save_model/$task/ckpt/$noise_ratio$noise_type
log_path=/data1/qd/noise_master/save_model/$task/log/${noise_ratio}${noise_type}_train.log
mkdir -p /data1/qd/noise_master/save_model/$task/ckpt
mkdir -p /data1/qd/noise_master/save_model/$task/log

CUDA_VISIBLE_DEVICES=$gpu_device python /home/qd/code/noise_master/train_base.py \
    --config  /home/qd/code/noise_master/config/default.config \
    --train_path $train_path \
    --model_dir $model_dir \
    --noise_ratio $noise_ratio \
    --noise_type $noise_type \
    --pick_ckpt best_acc > $log_path