# Stardard Libraries
import json
import os
import time

# Third party libraries
from tqdm import tqdm

# Self-defined Modules
from config import Config
from evidence_pattern_retrieval.ep_construction import EPCombiner, Combination
from my_utils.ap_utils import *
from my_utils.data_item import load_ds_items, DataItem
from my_utils.ep_utils import *
from my_utils.io_utils import read_json, write_json, append_jsonl, run_multiprocess
from my_utils.logger import Logger


logger = Logger.get_logger("ep_ranker_td_gen")


def get_quality_score(comb: Combination, gold_ans: List[str]) -> float:
    var_insts_map = comb.get_instantiated_results()
    insts = set()
    quality = 0
    # no terminal var
    if comb.terminal_var is None:
        for var in var_insts_map:
            insts |= set(var_insts_map[var])
        ans_cover_cnt = len(set(gold_ans) & set(insts))
        quality = ans_cover_cnt
    # terminal var (ans)
    else:
        insts = var_insts_map[comb.terminal_var]
        ans_cover_cnt = len(set(gold_ans) & set(insts))
        if len(insts) > 0 and ans_cover_cnt / len(insts) >= 0.1:
            quality = ans_cover_cnt
        else:
            quality = 0
    return quality


def td_gen_by_ans_coverage(
    ds_item: DataItem,
    ent_rels: Dict[str, Dict[str, List[str]]],
    rel_rels: Dict[str, Dict[str, List[str]]],
) -> dict:
    qid = ds_item.id
    text = ds_item.question
    ans = ds_item.answers
    if len(ent_rels) == 0:
        combs = []
        combs_wo_check = []
    else:
        combs = EPCombiner.combine(ent_rels, rel_rels, True)
        combs_wo_check = EPCombiner.combine(ent_rels, rel_rels, False)
    # calculate quality score for each combined pattern
    combs_with_quality = []
    max_quality = 0
    for comb in combs:
        quality = get_quality_score(comb, ans)
        max_quality = max(max_quality, quality)
        combs_with_quality.append((comb, quality))
    # collect combination strs with max quality score
    comb_strs_with_max_quality = set()
    for item in combs_with_quality:
        if max_quality < 0.01:
            break
        if item[1] == max_quality:
            comb_strs_with_max_quality.add(item[0].get_query_trips())
    # split positive & negative samples for training
    positives = []
    negatives = []
    for comb in combs_wo_check:
        comb_str = comb.get_query_trips()
        if comb_str in comb_strs_with_max_quality:
            positives.append(comb_str)
        else:
            negatives.append(comb_str)
    assert len(positives) == len(comb_strs_with_max_quality)
    ans = {"id": qid, "question": text, "positive_eps": positives, "negative_eps": negatives}
    return ans


def mp_td_gen_by_ans_coverage(target_file, pid, queue):
    while True:
        info = queue.get()
        if info is None:
            break
        ds_item = info["item"]
        ent_rels = info["er_aps"]
        rel_rels = info["rr_aps"]
        td_info = td_gen_by_ans_coverage(ds_item, ent_rels, rel_rels)
        append_jsonl(td_info, target_file)
        logger.debug(f"[mp_td_gen_by_ans_coverage] pid:{pid}, qid:{ds_item.id}")


def collect_train_data(split_tag: str, ap_topk: int, use_multi_process: bool):
    print(f">>> collect train data for ep ranking ({split_tag})...")
    assert split_tag in set(["dev", "train"])
    print("Reading dataset & retrieved aps")
    if split_tag == "dev":
        items = load_ds_items(Config.ds_dev)
    else:
        items = load_ds_items(Config.ds_train)
    retrieved_aps = filter_topk_aps(
        read_json(Config.retrieved_ap_f(split_tag)), ap_topk
    )
    td_f = Config.ep_rank_td_f(split_tag)
    print("Collect unprocessed items")
    processed_ids = set()
    if os.path.exists(td_f):
        with open(td_f) as f:
            for line in f:
                processed_ids.add(json.loads(line)["id"])
    logger.info(
        f"{len(processed_ids)} dev items already processed, {len(items)-len(processed_ids)} items left."
    )
    items_to_deal = []
    for item, ap_info in zip(items, retrieved_aps):
        assert item.id == ap_info["id"]
        if item.id not in processed_ids:
            er_aps, rr_aps = get_retrieved_aps(item.topic_ents, ap_info)
            items_to_deal.append({"item": item, "er_aps": er_aps, "rr_aps": rr_aps})
    print(f"Generate training data, use_multi_process: {use_multi_process}")
    if use_multi_process:
        run_multiprocess(
            mp_td_gen_by_ans_coverage, [td_f], items_to_deal, Config.worker_num
        )
    else:
        for info in tqdm(items_to_deal):
            td_info = td_gen_by_ans_coverage(info["item"], info["er_aps"], info["rr_aps"])
            append_jsonl(td_info, td_f)
            logger.debug(f"[td_gen_by_ans_coverage] qid:{info['item'].id}")
    logger.info(f"{split_tag} ep ranking training data generation done.")


def generate_candidate_eps(
    data_items: List[DataItem], ranked_ap_info: List[dict], ap_topk: int
) -> List[dict]:
    print(f"[Generate Candidate EPs] {len(data_items)} items total, ap_topk: {ap_topk}")
    ans = []
    retrieved_aps = filter_topk_aps(ranked_ap_info, ap_topk)
    assert len(data_items) == len(retrieved_aps)
    for item, ap_info in tqdm(zip(data_items, retrieved_aps)):
        assert item.id == ap_info["id"]
        er_aps, rr_aps = get_retrieved_aps(item.topic_ents, ap_info)
        if len(er_aps) == 0:
            combs = []
        else:
            combs = EPCombiner.combine(er_aps, rr_aps, False)
        ans.append(
            {
                "id": item.id,
                "question": item.question,
                "candidate_eps": [comb.get_query_trips() for comb in combs],
            }
        )
    return ans


def generate_candidate_eps_with_time_info(
    items: List[DataItem], split_tag: str, topk_range, step: int = 20
):
    topk_time_info = dict()
    topk_time_wo_io = dict()
    raw_topk = Config.ap_topk
    topks = range(topk_range[0], topk_range[1] + 1, step)
    ranked_ap_info = read_json(Config.retrieved_ap_f(split_tag))
    for topk in topks:
        Config.ap_topk = topk
        start = time.time()
        start_wo_io = time.time()
        candi_eps = generate_candidate_eps(items, ranked_ap_info, Config.ap_topk)
        end_wo_io = time.time()
        write_json(candi_eps, Config.candi_ep_f(split_tag))
        end = time.time()
        topk_time_info[f"top{topk}"] = round(end - start, 1)
        topk_time_wo_io[f"top{topk}"] = round(end_wo_io - start_wo_io, 1)
    Config.ap_topk = raw_topk
    # print(f"Time Costs:{topk_time_info}")
    print(f"Time Costs (w/o IO):{topk_time_wo_io}")


if __name__ == "__main__":
    if not os.path.exists(Config.ep_retrieval_dir):
        os.makedirs(Config.ep_retrieval_dir)
    Config.ap_topk = 100
    collect_train_data("dev", Config.ap_topk, True)
    collect_train_data("train", Config.ap_topk, True)
    Config.ap_topk = 100
    topk_range = [80, 100]
    step = 20
    generate_candidate_eps_with_time_info(load_ds_items(Config.ds_test), 'test', topk_range, step)
    topk_range = [100, 100]
    generate_candidate_eps_with_time_info(load_ds_items(Config.ds_dev), 'dev', topk_range, step)
    generate_candidate_eps_with_time_info(load_ds_items(Config.ds_train), 'train', topk_range, step)
