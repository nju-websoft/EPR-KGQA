#! /bin/bash
ds_tag=$1
split=$2
epoch=$3
topk=$4

sample_size=100
if [ "${ds_tag}" = "CWQ" ];then
    sample_size=64
fi

predict_file="data/${ds_tag}/ep_retrieval/training_data/${ds_tag}_${split}_top${topk}_ap_candi_eps_feature_cache_for_prediction"

model_dir="data/${ds_tag}/ep_retrieval/model/epoch_${epoch}"
if [ ! -d "${model_dir}" ];then
echo "model doesn't exist!"
exit
fi

python evidence_pattern_retrieval/ep_ranking.py --dataset ${ds_tag} --model_type bert --model_name_or_path ${model_dir} --do_lower_case --do_predict --predict_file ${predict_file} --overwrite_output_dir --max_seq_length 256 --output_dir data/${ds_tag}/ep_retrieval/ --per_gpu_eval_batch_size 16 --sample_size ${sample_size} --topk ${topk}