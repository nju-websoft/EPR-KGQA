# Stardard Libraries
import json
import os

# Third party libraries
from tqdm import tqdm

# Self-defined Modules
from config import Config
from evidence_pattern_retrieval.ep_construction import EPCombiner, Combination
from my_utils.ap_utils import *
from my_utils.data_item import load_ds_items, DataItem
from my_utils.ep_utils import *
from my_utils.io_utils import read_json, append_jsonl, run_multiprocess
from my_utils.logger import Logger


logger = Logger.get_logger("ep_ranker_td_gen")


def get_quality_score(comb: Combination, gold_ans: List[str]) -> float:
    var_insts_map = comb.get_instantiated_results()
    insts = set()
    quality = 0
    # no terminal var
    if comb.terminal_var == None:
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
    ans = {"id": qid, "question": text, "positive": positives, "negative": negatives}
    return ans


finished_td_gen = 0


def mp_td_gen_by_ans_coverage(target_file, pid, queue):
    while True:
        info = queue.get()
        if info == None:
            break
        ds_item = info["item"]
        ent_rels = info["ep_ap"]
        rel_rels = info["pp_ap"]
        td_info = td_gen_by_ans_coverage(ds_item, ent_rels, rel_rels)
        append_jsonl(td_info, target_file)
        global finished_td_gen
        finished_td_gen += 1
        print("finished td gen:", finished_td_gen)
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
            ep_aps, pp_aps = get_retrieved_aps(item.topic_ents, ap_info)
            items_to_deal.append({"item": item, "ep_ap": ep_aps, "pp_ap": pp_aps})
    print(f"Generate training data, use_multi_process: {use_multi_process}")
    if use_multi_process:
        run_multiprocess(
            mp_td_gen_by_ans_coverage, [td_f], items_to_deal, Config.worker_num
        )
    else:
        ### [To Del] for temp analysis
        count = 0
        pos_count = 0
        neg_count = 0
        fail_count = 0
        ### [To Del]
        for info in tqdm(items_to_deal):
            td_info = td_gen_by_ans_coverage(info["item"], info["ep_ap"], info["pp_ap"])
            append_jsonl(td_info, td_f)
            logger.debug(f"[td_gen_by_ans_coverage] qid:{info['item'].id}")
            ### [To Del] for temp analysis
            count += 1
            pos_count += len(td_info["positive"])
            neg_count += len(td_info["negative"])
            if len(td_info["positive"]) == 0:
                fail_count += 1
            if count == 100:
                print(
                    f"#item:{count}, #pos:{pos_count}, #neg:{neg_count}, #fail:{fail_count}"
                )
                break
            ### [To Del]
    logger.info(f"{split_tag} ep ranking training data generation done.")


def run():
    collect_train_data("dev", Config.ap_topk, True)
    collect_train_data("train", Config.ap_topk, True)


if __name__ == "__main__":
    collect_train_data("train", Config.ap_topk, True)
