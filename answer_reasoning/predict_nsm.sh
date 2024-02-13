ds_tag=$1
num_step=2
num_epoch=50

if [ "${ds_tag}" = "CWQ" ]
then
  num_step=4
  num_epoch=30
fi

CUDA_VISIBLE_DEVICES=0 python main_nsm.py --name ${ds_tag} --model_name gnn --data_folder ./datasets/${ds_tag}_EPR/ --checkpoint_dir checkpoint/pretrain/ --batch_size 20 --test_batch_size 40 --num_step ${num_step} --entity_dim 50 --word_dim 300 --kg_dim 100 --kge_dim 100 --eval_every 2 --experiment_name ${ds_tag}_nsm --eps 0.95 --num_epoch ${num_epoch} --use_self_loop --lr 5e-4 --q_type seq --word_emb_file word_emb_300d.npy --reason_kb --encode_type --loss_type kl --is_eval --load_experiment ${ds_tag}_nsm-f1.ckpt