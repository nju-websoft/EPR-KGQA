# Stardard Libraries
import time
from typing import *

# Third party libraries
from tqdm import tqdm

# Self-defined Modules
from config import Config
from evidence_pattern_retrieval.ep_construction import EPCombiner
from my_utils.ap_utils import filter_topk_aps, parse_snippet_dict_from_strs
from my_utils.data_item import DataItem, load_ds_items
from my_utils.ep_utils import *
from my_utils.io_utils import *


def snippet_dict_to_str_list(snippet_dict) -> List[str]:
    ans = []
    for item1 in snippet_dict:
        for tag in snippet_dict[item1]:
            for item2 in snippet_dict[item1][tag]:
                ans.append(f"{item1[3::]} {tag} {item2[3::]}")
    return ans


def generate_candidate_eps(
    data_items: List[DataItem], ranked_ap_info: List[dict], ap_topk: int
) -> List[dict]:
    print(f"[Generate Candidate Eps] {len(data_items)} items total, ap_topk: {ap_topk}")
    ans = []
    retrieved_aps = filter_topk_aps(ranked_ap_info, ap_topk)
    assert len(data_items) == len(retrieved_aps)
    for item, ap_info in tqdm(zip(data_items, retrieved_aps)):
        assert item.id == ap_info["id"]
        ep_aps, pp_aps = get_retrieved_aps(item.topic_ents, ap_info)
        if len(ep_aps) == 0:
            combs = []
        else:
            combs = EPCombiner.combine(ep_aps, pp_aps, False)
        ans.append(
            {
                "id": item.id,
                "question": item.question,
                "candidates": [comb.get_query_trips() for comb in combs],
            }
        )
    return ans


def induce_subg_from_topk_eps_in_batch(
    data_items: List[DataItem], ranked_ep_info: List[dict], topk=Config.ep_topk
) -> List[dict]:
    print(
        f"[Induce Subgraph] {len(data_items)} items total, ap_topk: {Config.ap_topk}, ep_topk: {Config.ep_topk}"
    )
    ranked_eps = []
    for info in ranked_ep_info:
        ranked_eps.append([pair[0] for pair in info["sorted_candidates_with_logits"]])
    ans = []
    assert len(data_items) == len(ranked_eps)
    for item, ranked_ep in tqdm(zip(data_items, ranked_eps)):
        qid = item.id
        question = item.question
        trips, node_ep_map, topk_eps = integrate_topk_instantiated_subg(ranked_ep, topk)
        ans.append(
            {
                "id": qid,
                "question": question,
                "answers": item.answers,
                "topics": item.topic_ents,
                "topk_ep": topk_eps,
                "subg": trips,
                "node_eps": node_ep_map,
            }
        )
    analyze_subg_info(data_items, ans)
    return ans

def analyze_subg_info(data_items: List[DataItem], instantiated_subgs: List[dict]):
    total = 0
    node_cnt = 0
    trip_cnt = 0
    ans_hit = 0
    for item, info in zip(data_items, instantiated_subgs):
        total += 1
        node_cnt += len(info["node_eps"])
        trip_cnt += len(info["subg"])
        if len(set(item.answers) & set(info["node_eps"].keys())) > 0:
            ans_hit += 1
    ans_hit = ans_hit / total
    node_avg = node_cnt / total
    trip_avg = trip_cnt / total
    print(
        f"total:{total}, avg #nodes:{node_avg}, avg #trips:{trip_avg}, ans hits:{ans_hit}"
    )


def analyze_subg_info_iid_setting(
    data_items: List[DataItem], instantiated_subgs: List[dict]
):
    total = 0
    node_cnt = 0
    trip_cnt = 0
    ans_hit = 0
    with open(Config.ds_test_iid_idxs, "r") as f:
        iid_idxs = set(json.load(f))
    idx = -1
    for item, info in zip(data_items, instantiated_subgs):
        idx += 1
        if idx not in iid_idxs:
            continue
        total += 1
        node_cnt += len(info["node_eps"])
        trip_cnt += len(info["subg"])
        if len(set(item.answers) & set(info["node_eps"].keys())) > 0:
            ans_hit += 1
    ans_hit = ans_hit / total
    node_avg = node_cnt / total
    trip_avg = trip_cnt / total
    print(
        f"total:{total}, avg #nodes:{node_avg}, avg #trips:{trip_avg}, ans hits:{ans_hit}"
    )


def analyze_subg_info_zero_shot_setting(
    data_items: List[DataItem], instantiated_subgs: List[dict]
):
    total = 0
    node_cnt = 0
    trip_cnt = 0
    ans_hit = 0
    with open(Config.ds_test_zero_shot_idxs, "r") as f:
        zero_shot_idxs = set(json.load(f))
    idx = -1
    for item, info in zip(data_items, instantiated_subgs):
        idx += 1
        if idx not in zero_shot_idxs:
            continue
        total += 1
        node_cnt += len(info["node_eps"])
        trip_cnt += len(info["subg"])
        if len(set(item.answers) & set(info["node_eps"].keys())) > 0:
            ans_hit += 1
    ans_hit = ans_hit / total
    node_avg = node_cnt / total
    trip_avg = trip_cnt / total
    print(
        f"total:{total}, avg #nodes:{node_avg}, avg #trips:{trip_avg}, ans hits:{ans_hit}"
    )


def generate_candidate_eps_with_time_info(
    items: List[DataItem], split_tag: str, topk_range: int = 200, step: int = 20
):
    topk_time_info = dict()
    topk_time_wo_io = dict()
    raw_topk = Config.ap_topk
    topks = range(topk_range[0], topk_range[1] + 1, step)
    for topk in topks:
        Config.ap_topk = topk
        start = time.time()
        ranked_ap_info = read_json(Config.retrieved_ap_f(split_tag))
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


def induce_subgraph_with_time_info(
    items: List[DataItem],
    split_tag: str,
    topk_range: List[int] = [20, 200],
    step: int = 20,
):
    topk_time_info = dict()
    topk_time_wo_io = dict()
    raw_topk = Config.ap_topk
    topks = range(topk_range[0], topk_range[1] + 1, step)
    for topk in topks:
        Config.ap_topk = topk
        start = time.time()
        ranked_ep_info = read_json(Config.ranked_ep_f(split_tag))
        start_wo_io = time.time()
        induced_subg = induce_subg_from_topk_eps_in_batch(
            items, ranked_ep_info, topk=Config.ep_topk
        )
        end_wo_io = time.time()
        # write_json(induced_subg, Config.induced_subg_f(split_tag))
        end = time.time()
        topk_time_info[f"top{topk}"] = round(end - start, 1)
        topk_time_wo_io[f"top{topk}"] = round(end_wo_io - start_wo_io, 1)
    Config.ap_topk = raw_topk
    # print(f"Time Costs: {topk_time_info}")
    print(f"Time Costs (w/o IO): {topk_time_wo_io}")


def calculate_CR(data_items, split_tag, topk_range, step):
    raw_topk = Config.ap_topk
    topks = range(topk_range[0], topk_range[1] + 1, step)
    for topk in topks:
        Config.ap_topk = topk
        subgs = read_json(Config.induced_subg_f(split_tag))
        print(topk, end=": ")
        analyze_subg_info(data_items, subgs)
    print("")
    for topk in topks:
        Config.ap_topk = topk
        subgs = read_json(Config.induced_subg_f(split_tag))
        print(topk, end=": ")
        analyze_subg_info_iid_setting(data_items, subgs)
    print("")
    for topk in topks:
        Config.ap_topk = topk
        subgs = read_json(Config.induced_subg_f(split_tag))
        print(topk, end=": ")
        analyze_subg_info_zero_shot_setting(data_items, subgs)
    Config.ap_topk = raw_topk


def run(
    split_filename: str,
    split_tag: str,
    topk_range: List[int] = [20, 200],
    step: int = 20,
):
    print(f"\n>>> run inference ({Config.ds_tag}.{split_tag})...")
    ds_items = load_ds_items(split_filename)

    # Step1: 生成检索到的 atomic patterns 候选
    # 【TODO】

    # Step2: 组合出候选 evidence patterns
    # generate_candidate_eps_with_time_info(ds_items, split_tag, topk_range, step)

    # Step3: 对候选 evidence patterns 进行排序
    # 【TODO】

    # Step4:
    induce_subgraph_with_time_info(ds_items, split_tag, topk_range, step)


if __name__ == "__main__":
    # run(Config.ds_dev, "dev", [20, 200], 20)
    run(Config.ds_test, "test", [100, 100], 20)
    # calculate_CR(load_ds_items(Config.ds_test), "test", [20, 200], 20)
