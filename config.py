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
    ds_split_f = None
    if ds_tag == 'CWQ':
        ds_split_f = lambda split: f"data/dataset/CWQ/ComplexWebQuestions_{split}.json"
    elif ds_tag == 'WebQSP':
        ds_split_f = lambda split: f"data/dataset/WebQSP/{split}_simple.jsonl"
    
    # cache
    cache_dir = f"data/cache/{ds_tag}"
    cache_rel_info = "data/cache/relation_info_fb.json"
    cache_type_info = "data/cache/type_info_fb.json"
    cache_rel_conn = "data/cache/rel_conn_fb.jsonl"
    cache_rr_aps = "data/cache/rr_aps_fb.json"
    cache_path = config["cache"]["searched_path"]
    
    # log
    log_dir = "data/log/"
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
    retrieved_ap_f = lambda split: f"{Config.ap_retrieval_dir}{split}_retrieved_ap.json"

    # ep generation
    ep_retrieval_dir = config["ep_retrieval"]["work_dir"]
    max_combine_rels = config["ep_retrieval"]["max_combine_rels"]
    ep_rank_td_f = lambda split: f"{Config.ep_retrieval_dir}{split}_top{Config.ap_topk}_ap_ep_rank_td.jsonl"
    candi_ep_f = lambda split: f"{Config.ep_retrieval_dir}{split}_top{Config.ap_topk}_ap_candi_ep.json"
    ranked_ep_f = lambda split: f"{Config.ep_retrieval_dir}{split}_top{Config.ap_topk}_ap_ranked_ep.json"

    # subgraph extraction
    allow_literal = True if ds_tag == 'WebQSP' else False
    subgraph_extraction_dir = config["subgraph_extraction"]["work_dir"]
    ep_topk = config["subgraph_extraction"]["ep_topk"]
    feature = lambda: f"top{Config.ap_topk}_ap_top{Config.ep_topk}_ep"
    ans_rank_td_f = lambda split: f"{Config.subgraph_extraction_dir}{Config.ds_tag}_{split}_{Config.feature()}_ans_rank_td.json"
    induced_subg_f = lambda split: f"{Config.subgraph_extraction_dir}{Config.ds_tag}_{split}_{Config.feature()}_instantiated_subg.json"
