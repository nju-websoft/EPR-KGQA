# Stardard Libraries
from copy import deepcopy
from typing import List, Set, Tuple

# Third party libraries
from tqdm import tqdm

# Self-defined Modules
from config import Config
from my_utils.data_item import DataItem, load_ds_items
from my_utils.freebase import FreebaseODBC
from my_utils.io_utils import append_jsonl, read_jsonl_by_key
from my_utils.logger import Logger
from my_utils.pred_base import PredBase


banned_base_preds = set(
    [
        "ns:type.type.instance",
        "ns:type.object.type",
        "ns:common.topic.notable_types",
        "ns:common.notable_for.notable_object",
    ]
)
logger = Logger.get_logger("Preprocess", True)


def heuristic_search_path(data_items: List[DataItem], tag: str):
    # 读取已经缓存的搜索结果
    cached_path = read_jsonl_by_key(Config.cache_path)
    freebase = FreebaseODBC()
    skip_count = 0
    fail_count = 0
    for item in tqdm(data_items):
        # 若问题的所有 Topic Entity 都成功搜索出至少一条路径则称做搜索成功
        search_succ = True
        topic_ents = set(item.topic_ents)
        ans_ents = item.answers
        key_lexicals = item.get_question_key_lexical()
        results = []
        if len(topic_ents) == 0 or len(ans_ents) == 0:
            skip_count += 1
            continue
        # 检查数据项是否在 cache 中
        if item.id in cached_path:
            results = cached_path[item.id]["paths"]
            searched_topic_ents = set()
            for path in results:
                searched_topic_ents.add(path[0])
            search_succ = len(topic_ents - searched_topic_ents) == 0
        else:
            # topic_ent -> ans_ent 正向搜索
            for topic_ent in topic_ents:
                paths = []
                forward_search(
                    topic_ent, set(ans_ents), [topic_ent], paths, freebase, key_lexicals
                )
                search_succ = search_succ and len(paths) > 0
                results += paths
            append_jsonl({"id": item.id, "paths": results}, Config.cache_path)
        if not search_succ:
            fail_count += 1
    logger.info(
        f"[heuristic_search_path]({tag}) total:{len(data_items)}, skip count:{skip_count}, \
fail count:{fail_count}, succ rate:{round(1-(skip_count+fail_count)/len(data_items), 4)}"
    )


# do topic->ans path search recursively (DFS manner)
def forward_search(
    topic_ent: str,
    ans_ents: Set[str],
    cur_path: List[str],
    all_paths: List[List[str]],
    freebase: FreebaseODBC,
    key_lexical: Set[str],
):
    # if path length exceeds the hop limit, then stop
    if len(cur_path) >= Config.max_hop_limit + 1:
        return
    # step1: obtain all the predicates in the one-hop neighborhood
    neighbor_preds = freebase.query_neighbor_preds(cur_path)
    binary_preds, nary_preds = pred_classification(neighbor_preds, cur_path)
    # step2: obtain predstr directly connected to the answer, if search succ, record path and return
    predstrs_to_ans = freebase.query_predstr_to_ans(cur_path, nary_preds, ans_ents)
    if len(predstrs_to_ans) > 0:
        for predstr in predstrs_to_ans:
            new_path = deepcopy(cur_path)
            new_path.append(predstr)
            all_paths.append(new_path)
    # step3: obtain n-ary predstrs related to question, then do search on a new base path
    else:
        nary_predstrs = freebase.query_nary_predstrs(cur_path, nary_preds)
        related_predstrs = filter_related_predstrs(
            nary_predstrs + binary_preds, key_lexical
        )
        for predstr in related_predstrs:
            new_path = deepcopy(cur_path)
            new_path.append(predstr)
            if satisfy_base_path_condition(new_path):
                forward_search(
                    topic_ent, ans_ents, new_path, all_paths, freebase, key_lexical
                )


def pred_classification(preds: List[str], base_path: List[str]) -> Tuple:
    binary_preds = []
    nary_preds = []
    for pred in preds:
        # 不能加入新的 base path 的谓词没有分类的意义
        if not satisfy_base_path_condition(deepcopy(base_path) + [pred]):
            continue
        if PredBase.reach_cvt(pred):
            nary_preds.append(pred)
        else:
            binary_preds.append(pred)
    return binary_preds, nary_preds


def filter_related_predstrs(predstrs: List[str], key_lexical: Set[str]) -> List[str]:
    related_predstr = []
    for pred_str in predstrs:
        if len(get_predstr_key_lexical(pred_str) & key_lexical) > 0:
            related_predstr.append(pred_str)
    return related_predstr


def get_predstr_key_lexical(pred_str: str) -> Set[str]:
    tokens = set()
    for pred in pred_str.split("..."):
        if pred.endswith("_Rev"):
            pred = pred[:-4]
        tokens |= PredBase.get_pred_keywords(pred)
    lemmas = DataItem.get_lemmas(tokens)
    return tokens | lemmas


def satisfy_base_path_condition(base_path: List[str]) -> bool:
    """任何可能生成新的 base path 的地方都需要通过这个检查"""
    # Rule1: Base Path 中不能出现实例转移到类型或从类型转移到实例的情况
    path_str = " ".join(base_path)
    for pred in banned_base_preds:
        if path_str.find(pred) != -1:
            return False
    # Rule2: Path 中不能存在直接往回走的情况(两个相邻的逆关系)
    preds = []
    for predstr in base_path:
        preds += predstr.split("...")
    for idx in range(len(preds))[1:]:
        if idx + 1 < len(preds):
            first = preds[idx]
            second = preds[idx + 1]
            if PredBase.is_reverse_pair(first, second):
                return False
    return True


def run(run_tag=Config.bulk_path_search):
    if not run_tag:
        return
    logger.info(">>> reading dataset...")
    train_set = load_ds_items(Config.ds_train)
    dev_set = load_ds_items(Config.ds_dev)
    logger.info(
        f"{len(train_set)} items in train set; {len(dev_set)} items in dev set;"
    )

    logger.info(">>> heuristic path searching...")
    heuristic_search_path(dev_set, "dev")
    heuristic_search_path(train_set, "train")
