CUDA_ID=$1
output_dir=$2
model_path=$(ls -d $output_dir*/)
base_model=/data/llmweights/llama-7b-hf
test_data=/home/wangshirui/llm/lora-poi/df_data/$3_constrast/test.json
embedding_weights=/home/wangshirui/Secor/df_data/$3/checkpoints/lgn-3-128.pth.tar
for path in $model_path
do
    echo $path
    CUDA_VISIBLE_DEVICES=$CUDA_ID python3 test_secor.py \
        --base_model $base_model \
        --lora_weights $path \
        --embedding_weights $embedding_weights \
        --test_data_path $test_data \
        --result_json_data $2.json \
        --batch_size 8 
done
