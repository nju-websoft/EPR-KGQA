#! /bin/bash
ds_tag=$1

sample_size=100
if [ "${ds_tag}" = "CWQ" ];then
    sample_size=64
fi

train_file="data/${ds_tag}/ep_retrieval/training_data/${ds_tag}_train_feature_cache_for_training_sample_size_${sample_size}"
predict_file="data/${ds_tag}/ep_retrieval/training_data/${ds_tag}_dev_feature_cache_for_training_sample_size_${sample_size}"

model_dir="data/${ds_tag}/ep_retrieval/model/"
if [ ! -d "${model_dir}" ];then
mkdir ${model_dir}
fi


CUDA_VISIBLE_DEVICES=1 python evidence_pattern_retrieval/ep_ranking.py  --dataset ${ds_tag} --model_type bert --model_name_or_path bert-base-uncased --do_lower_case --do_train --do_eval --disable_tqdm --train_file ${train_file} --predict_file ${predict_file} --learning_rate 1e-5 --evaluate_during_training --num_train_epochs 10 --overwrite_output_dir --max_seq_length 256 --logging_steps 200 --eval_steps 100000000 --save_steps 100000000 --warmup_ratio 0.1 --output_dir ${model_dir} --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 2 --sample_size ${sample_size} --topk 100