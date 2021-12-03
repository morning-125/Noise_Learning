gpu_device=7

train_path=/data1/lxb/qd/noise_master/ag_news/ag_news_csv/train.csv

task=cl
noise_type=asym
noise_ratio=0.3

model_dir=/data1/lxb/qd/noise_master/save_model/$task/ckpt/$noise_ratio$noise_type
log_path=/data1/lxb/qd/noise_master/save_model/$task/log/${noise_ratio}${noise_type}_train.log
mkdir -p /data1/lxb/qd/noise_master/save_model/$task/ckpt
mkdir -p /data1/lxb/qd/noise_master/save_model/$task/log

CUDA_VISIBLE_DEVICES=$gpu_device python /home/lxb/qd/noise/Noise_Learning/train_cl.py \
    --config  /home/lxb/qd/noise/Noise_Learning/config/default.config \
    --train_path $train_path \
    --model_dir $model_dir \
    --noise_ratio $noise_ratio \
    --noise_type $noise_type \
    --pick_ckpt best_acc > $log_path