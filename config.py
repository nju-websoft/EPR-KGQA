import yaml
import argparse


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_CWQ.yaml", help="experimental configuration file")
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

    # cache
    cache_pred_info = "data/cache/predicate_info_fb.json"
    cache_type_info = "data/cache/type_info_fb.json"
    cache_pred_conn = "data/cache/pred_conn_fb.jsonl"
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
    ap_retrieve_td_f = lambda split: f"{Config.ap_retrieval_dir}{split}_ap_retrieve_td.json"
    retrieved_ap_f = lambda split: f"{Config.ap_retrieval_dir}{split}_ranked_ap.json"

    # ep generation
    ep_generation_dir = config["ep_generation"]["work_dir"]
    max_combine_preds = config["ep_generation"]["max_combine_preds"]
    ep_rank_td_f = lambda split: f"{Config.ep_generation_dir}{split}_top{Config.ap_topk}_ap_ep_rank_td.jsonl"
    candi_ep_f = lambda split: f"{Config.ep_generation_dir}{split}_top{Config.ap_topk}_ap_candi_ep.json"
    ranked_ep_f = lambda split: f"{Config.ep_generation_dir}{split}_top{Config.ap_topk}_ap_ranked_ep.json"

    # ans selection
    ans_selection_dir = config["ans_selection"]["work_dir"]
    ep_topk = config["ans_selection"]["ep_topk"]
    feature = lambda: f"top{Config.ap_topk}_ap_top{Config.ep_topk}_ep"
    ans_rank_td_f = lambda split: f"{Config.ans_selection_dir}{Config.ds_tag}_{split}_{Config.feature()}_ans_rank_td.json"
    induced_subg_f = lambda split: f"{Config.ans_selection_dir}{Config.ds_tag}_{split}_{Config.feature()}_instantiated_subg.json"
