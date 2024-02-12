# Stardard Libraries
import time
from typing import List

# Third party libraries
from tqdm import tqdm
from tqdm.contrib import tzip

# Self-defined Modules
from config import Config
from my_utils.data_item import DataItem, load_ds_items
from my_utils.ep_utils import integrate_topk_instantiated_subg, get_instantiable_topk
from my_utils.io_utils import *
from my_utils.logger import Logger


logger = Logger.get_logger("ans_ranker_td_gen", True)


def td_gen(ds_items: List[DataItem], ranked_eps: List[List[str]], topk=Config.ep_topk):
    td = []
    assert len(ds_items) == len(ranked_eps)
    for item, eps in tzip(ds_items, ranked_eps):
        qid = item.id
        question = item.question
        answers = item.answers
        # 先做一遍可实例化过滤再进行实例化的执行速度更快
        topk_eps = get_instantiable_topk(eps, topk)
        trips, node_ep_map, _ = integrate_topk_instantiated_subg(topk_eps, topk)
        td.append(
            {
                "id": qid,
                "question": question,
                "answers": answers,
                "topics": item.topic_ents,
                "topk_ep": topk_eps,
                "subg": trips,
                "node_eps": node_ep_map,
            }
        )
        # break
    return td


def analyze_td(items: List[DataItem], td_f: str, logger):
    td = read_json(td_f)
    total = 0
    node_cnt = 0
    trip_cnt = 0
    ans_hit = 0
    for item, info in zip(items, td):
        total += 1
        node_cnt += len(info["node_eps"])
        trip_cnt += len(info["subg"])
        if len(set(item.answers) & set(info["node_eps"].keys())) > 0:
            ans_hit += 1
    ans_hit = ans_hit / total
    node_avg = node_cnt / total
    trip_avg = trip_cnt / total
    logger.info(
        f"total:{total}, avg #nodes:{node_avg}, avg #trips:{trip_avg}, ans hits:{ans_hit}"
    )


def generate_subgraph_for_training():
    # 验证集
    if os.path.exists(Config.ans_rank_td_f("dev")):
        print(f"{Config.ans_rank_td_f('dev')} already exists!")
    else:
        dev_items = load_ds_items(Config.ds_dev)
        dev_ranked_eps = read_json_list_by_key(Config.ranked_ep_f("dev"), "id")
        temp = []
        for item in dev_items:
            info = dev_ranked_eps[item.id]
            temp.append([pair[0] for pair in info["sorted_candidates_with_logits"]])
        dev_ranked_eps = temp
        dev_td = td_gen(dev_items, dev_ranked_eps, Config.ep_topk)
        write_json(dev_td, Config.ans_rank_td_f("dev"))
        analyze_td(dev_items, Config.ans_rank_td_f("dev"), logger)

    # 训练集
    if os.path.exists(Config.ans_rank_td_f("train")):
        print(f"{Config.ans_rank_td_f('train')} already exists!")
    else:
        train_items = load_ds_items(Config.ds_train)
        train_ranked_eps = read_json_list_by_key(Config.ranked_ep_f("train"), "id")
        temp = []
        for item in train_items:
            info = train_ranked_eps[item.id]
            temp.append([pair[0] for pair in info["sorted_candidates_with_logits"]])
        train_ranked_eps = temp
        train_td = td_gen(train_items, train_ranked_eps, Config.ep_topk)
        write_json(train_td, Config.ans_rank_td_f("train"))
        analyze_td(train_items, Config.ans_rank_td_f("train"), logger)


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
    for item, ranked_ep in tzip(data_items, ranked_eps):
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
        write_json(induced_subg, Config.induced_subg_f(split_tag))
        end = time.time()
        topk_time_info[f"top{topk}"] = round(end - start, 1)
        topk_time_wo_io[f"top{topk}"] = round(end_wo_io - start_wo_io, 1)
    Config.ap_topk = raw_topk
    print(f"Time Costs: {topk_time_info}")
    print(f"Time Costs (w/o IO): {topk_time_wo_io}")


if __name__ == "__main__":
    if not os.path.exists(Config.subgraph_extraction_dir):
        os.makedirs(Config.subgraph_extraction_dir)
    # extract subgraph for dev and train
    Config.ap_topk = 100
    Config.ep_topk = 3
    generate_subgraph_for_training()
    # extract subgraph for test
    Config.ep_topk = 1
    ds_items = load_ds_items(Config.ds_test)
    induce_subgraph_with_time_info(ds_items, "test", [80, 100], 20)
