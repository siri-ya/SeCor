# $1 Device ID, $2 Random Seed, $3 Dataset
echo $1, $2, $3
seed=$2
output_dir=yourdirectory/SeCor_$3_$2
base_model=yourdirectory/llama-2-7b-hf
cf_model=yourdirectory/lgn-3-128.pth.tar
# visit history data
data=yourdirectory/train.json
# POI description data
description_path=yourdirectory/description_train.npy
val_size=8000
instruction_model=None
for lr in 1e-4
do
    for sample in -1
    do
            mkdir -p $output_dir
            echo "lr: $lr, dropout: $dropout, seed: $seed, sample: $sample"
            CUDA_VISIBLE_DEVICES=$1 python3 -u finetune_secor.py \
                --base_model $base_model \
                --cf_model $cf_model \
                --data_path $data \
                --description_path $description_path \
                --val_set_size $val_size \
                --output_dir ${output_dir}_${seed}_${sample} \
                --batch_size 64 \
                --micro_batch_size 16 \
                --num_epochs 50 \
                --learning_rate $lr \
                --cutoff_len 1024 \
                --lora_r 8 \
                --lora_alpha 16 \
                --lora_dropout 0.05 \
                --lora_target_modules '[q_proj,v_proj]' \
                --resume_from_checkpoint $instruction_model \
                --sample $sample \
                --seed $2 
    done
done
