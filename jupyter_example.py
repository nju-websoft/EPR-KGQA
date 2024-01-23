# ！！！ need restart kernel to refresh import
# 批量运行 topk lp 对 NSM 结果的影响（WebQSP）
ds_tag = "WebQSP"
num_step = 2
num_epoch = 50
import sys

sys.argv = ['config.py', '--config', f'config_{ds_tag}.yaml']
import os

os.environ['MKL_THREADING_LAYER'] = 'GNU'
from trans_to_nsm_input import trans_to_nsm_format, load_itemid_map, read_json, Config

nsm_new_dir = f"/home3/jmli/projects/NSM_H/datasets/{ds_tag}_EGPSR/"
ours_nsm_dir = f"/home3/jmli/projects/IRQA/data/dataset/{ds_tag}_NSM/"

entity_file = f"/home3/jmli/projects/IRQA/data/dataset/{ds_tag}_NSM/entities.txt"
relation_file = f"/home3/jmli/projects/IRQA/data/dataset/{ds_tag}_NSM/relations.txt"
ours_nsm_subg_file = lambda tag: f"/home3/jmli/projects/IRQA/data/dataset/{ds_tag}_NSM/{tag}_simple.json"

ent2id = load_itemid_map(entity_file)
rel2id = load_itemid_map(relation_file)

# 替换测试集导出子图文件，并进行一一评测
res = []
for topk in range(20, 210, 20):
    Config.pp_topk = topk
    Config.ep_topk = 1
    test_topk_subg = read_json(Config.induced_subg_f("test"))
    trans_to_nsm_format(ent2id, rel2id, test_topk_subg, ours_nsm_subg_file("test"))
    os.system('cp {ours_nsm_dir}test_simple.json {nsm_new_dir}test_simple.json')
    os.chdir("/home3/jmli/projects/NSM_H/")

    # best_f1_out = os.system(f'CUDA_VISIBLE_DEVICES=1 python main_nsm.py --name {ds_tag} --model_name gnn --data_folder ./datasets/{ds_tag}_EGPSR/ --checkpoint_dir checkpoint/pretrain/ --batch_size 20 --test_batch_size 40 --num_step {num_step} --entity_dim 50 --word_dim 300 --kg_dim 100 --kge_dim 100 --eval_every 2 --experiment_name {ds_tag}_nsm --eps 0.95 --num_epoch {num_epoch} --use_self_loop --lr 5e-4 --q_type seq --word_emb_file word_emb_300d.npy --reason_kb --encode_type --loss_type kl --is_eval --load_experiment {ds_tag}_nsm-f1.ckpt')

    # best_h1_out = ! CUDA_VISIBLE_DEVICES=1 python main_nsm.py --name {ds_tag} --model_name gnn --data_folder ./datasets/{ds_tag}_EGPSR/ --checkpoint_dir checkpoint/pretrain/ --batch_size 20 --test_batch_size 40 --num_step {num_step} --entity_dim 50 --word_dim 300 --kg_dim 100 --kge_dim 100 --eval_every 2 --experiment_name {ds_tag}_nsm --eps 0.95 --num_epoch {num_epoch} --use_self_loop --lr 5e-4 --q_type seq --word_emb_file word_emb_300d.npy --reason_kb --encode_type --loss_type kl --is_eval --load_experiment {ds_tag}_nsm-h1.ckpt

    os.chdir("/home3/jmli/projects/IRQA/")

    # print(f"[test] lp top{topk}, ep top1: (Best F1) {best_f1_out[-1]}; (Best H1) {best_h1_out[-1]}")
    # print(f"[{ds_tag} Test] lp top{topk}, ep top1: (best_f1_model) {best_f1_out[-1]}")