import yaml
import argparse

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config_CWQ.yaml", help="experimental configuration file")
    args = parser.parse_args()
    return args

class Config:
    args = _parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)
        print(f"CONFIG: {f.name}")
    
    # dataset 
    ds_tag = config["dataset"]["tag"]
    ds_train = config["dataset"]["train"]
    ds_dev = config["dataset"]["dev"]
    ds_test = config["dataset"]["test"]
    ds_sr_all = config["dataset"]["SR_cache_all"]
    ds_train_relations = config["dataset"]["train_relations"]
    ds_test_iid_idxs = config["dataset"]["test_iid_idxs"]
    ds_test_zero_shot_idxs = config["dataset"]["test_zero_shot_idxs"]
    
    # cache
    cache_pred_info = config["cache"]["pred_info"]
    cache_type_info = config["cache"]["type_info"]
    cache_pred_conn = config["cache"]["pred_conn"]
    cache_path = config["cache"]["searched_path"]
    
    # log
    log_process = config["log"]["process_log"]
    log_level = config["log"]["level"]
    
    # global var
    query_time_limit = config["global_var"]["query_time_limit"]
    max_hop_limit = config["global_var"]["max_hop_limit"]
    bulk_path_search = config["global_var"]["bulk_path_search"]
    worker_num = config["global_var"]["worker_num"]
    
    # ap retrieval
    ap_retrieval_dir = config["ap_retrieval"]["work_dir"]
    ap_topk = config["ap_retrieval"]["ap_topk"]
    ap_rank_td_f = lambda tag : f"{Config.ap_retrieval_dir}{tag}_lp_rank_td.json"
    retrieved_ap_f = lambda tag : f"{Config.ap_retrieval_dir}{tag}_ranked_lp.json"
    
    # ep generation
    ep_generation_dir = config["ep_generation"]["work_dir"]
    max_combine_preds = config["ep_generation"]["max_combine_preds"]
    ep_rank_td_f = lambda tag : f"{Config.ep_generation_dir}{tag}_top{Config.ap_topk}_lp_ep_rank_td.jsonl"
    candi_ep_f = lambda tag : f"{Config.ep_generation_dir}{tag}_top{Config.ap_topk}_lp_candi_ep.json"
    ranked_ep_f = lambda tag : f"{Config.ep_generation_dir}{tag}_top{Config.ap_topk}_lp_ranked_ep.json"
    
    # ans selection
    ans_selection_dir = config["ans_selection"]["work_dir"]
    ep_topk = config["ans_selection"]["ep_topk"]
    feature = lambda : f"top{Config.ap_topk}_lp_top{Config.ep_topk}_ep"
    ans_rank_td_f = lambda tag : f"{Config.ans_selection_dir}{Config.ds_tag}_{tag}_{Config.feature()}_ans_rank_td.json"
    induced_subg_f = lambda tag : f"{Config.ans_selection_dir}{Config.ds_tag}_{tag}_{Config.feature()}_instantiated_subg.json"
    ranked_ans_f = lambda tag : f"{Config.ans_selection_dir}{Config.ds_tag}_{tag}_{Config.feature()}_ranked_ans.json"