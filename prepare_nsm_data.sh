ds_tag=$1
num_step=2
num_epoch=50

if [ "${ds_tag}" = "CWQ" ]
then
  num_step=4
  num_epoch=30
fi

nsm_raw_dir="/home3/jmli/projects/NSM_H/datasets/${ds_tag}/"
nsm_new_dir="/home3/jmli/projects/NSM_H/datasets/${ds_tag}_EGPSR/"
ours_nsm_dir="/home3/jmli/projects/IRQA/data/dataset/${ds_tag}_NSM/"

echo ">>> preparing NSM traning data (${ds_tag})"

# python trans_to_nsm_input.py --config "config_${ds_tag}.yaml"

# rm -rf ${nsm_new_dir}
# cp -r ${nsm_raw_dir} ${nsm_new_dir}

# f_names='dev_simple.json test_simple.json train_simple.json entities.txt relations.txt'

# for f in ${f_names}; do
#   cp "${ours_nsm_dir}${f}" "${nsm_new_dir}${f}"
# done

# echo ">>> training NSM model (${ds_tag})"

cd /home3/jmli/projects/NSM_H/

# CUDA_VISIBLE_DEVICES=1 python main_nsm.py --model_name gnn --data_folder ./datasets/${ds_tag}_EGPSR/ --checkpoint_dir checkpoint/pretrain/ --batch_size 20 --test_batch_size 40 --num_step ${num_step} --entity_dim 50 --word_dim 300 --kg_dim 100 --kge_dim 100 --eval_every 2 --encode_type --experiment_name ${ds_tag}_nsm --eps 0.95 --num_epoch ${num_epoch} --use_self_loop --lr 5e-4 --q_type seq --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb

CUDA_VISIBLE_DEVICES=3 python main_nsm.py --name ${ds_tag} --model_name gnn --data_folder ./datasets/${ds_tag}_EGPSR/ --checkpoint_dir checkpoint/pretrain/ --batch_size 20 --test_batch_size 40 --num_step ${num_step} --entity_dim 50 --word_dim 300 --kg_dim 100 --kge_dim 100 --eval_every 2 --experiment_name ${ds_tag}_nsm --eps 0.95 --num_epoch ${num_epoch} --use_self_loop --lr 5e-4 --q_type seq --word_emb_file word_emb_300d.npy --reason_kb --encode_type --loss_type kl --is_eval --load_experiment ${ds_tag}_nsm-f1.ckpt