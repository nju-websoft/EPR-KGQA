#! /bin/bash
ds_tag=$1
train_file="data/${ds_tag}/ap_retrieval/training_data/${ds_tag}_train_feature_cache_sample_size_20"
dev_file="data/${ds_tag}/ap_retrieval/training_data/${ds_tag}_dev_feature_cache_sample_size_20"
model_dir="data/${ds_tag}/ap_retrieval/model/"

if [ ! -d "${model_dir}" ];then
mkdir ${model_dir}
fi

CUDA_VISIBLE_DEVICES=0 python atomic_pattern_retrieval/biencoder/run_biencoder.py --dataset_type ${ds_tag} --train_file ${train_file} --dev_file ${dev_file} --model_save_path ${model_dir} --batch_size 16 --epochs 5 --log_dir ${model_dir} --cache_dir bert-base-uncased