# Stardard Libraries
from typing import List

# Third party libraries
from tqdm import tqdm

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
    for item, eps in tqdm(zip(ds_items, ranked_eps)):
        qid = item.id
        question = item.question
        answers = item.answers
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


# WARN: 实践证明没有先做一遍可实例化过滤再进行实例化的执行速度快
def td_gen_wo_pre_inst_check(
    ds_items: List[DataItem], ranked_eps: List[List[str]], topk=Config.ep_topk
):
    td = []
    assert len(ds_items) == len(ranked_eps)
    for item, eps in tqdm(zip(ds_items, ranked_eps)):
        qid = item.id
        question = item.question
        answers = item.answers
        trips, node_ep_map, topk_eps = integrate_topk_instantiated_subg(eps, topk)
        td.append(
            {
                "id": qid,
                "question": question,
                "answers": answers,
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


def run():
    # 验证集
    dev_items = load_ds_items(Config.ds_dev)
    dev_ranked_eps = read_json_list_by_key(Config.ranked_ep_f("dev"), "ID")
    temp = []
    for item in dev_items:
        info = dev_ranked_eps[item.id]
        temp.append([pair[0] for pair in info["sorted_candidates_with_logits"]])
    dev_ranked_eps = temp
    dev_td = td_gen(dev_items, dev_ranked_eps, Config.ep_topk)
    write_json(dev_td, Config.ans_rank_td_f("dev"))
    analyze_td(dev_items, Config.ans_rank_td_f("dev"), logger)

    # 训练集
    train_items = load_ds_items(Config.ds_train)
    train_ranked_eps = read_json_list_by_key(Config.ranked_ep_f("train"), "ID")
    temp = []
    for item in train_items:
        info = train_ranked_eps[item.id]
        temp.append([pair[0] for pair in info["sorted_candidates_with_logits"]])
    train_ranked_eps = temp
    train_td = td_gen(train_items, train_ranked_eps, Config.ep_topk)
    write_json(train_td, Config.ans_rank_td_f("train"))
    analyze_td(train_items, Config.ans_rank_td_f("train"), logger)
