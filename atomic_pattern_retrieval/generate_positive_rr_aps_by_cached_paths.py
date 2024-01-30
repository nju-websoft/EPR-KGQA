# Stardard Libraries
from copy import deepcopy
import os
from typing import Dict, List, Tuple

# Third party libraries
from tqdm import tqdm

# Self-defined Modules
from config import Config
from my_utils.ap_utils import accumulate_rel_rel_statistics, cover_gold_rr_aps, \
    normalize_triplet, parse_triplets_from_serialized_ep
from my_utils.data_item import load_ds_items
from my_utils.fact import Fact
from my_utils.io_utils import write_json, read_jsonl_by_key
from preprocess.adjacent_info_prepare import AdjacentInfoPrepare


def get_rr_aps_from_searched_path(paths: List[List[str]], topic_ents: List[str]) -> dict:
    """
        1. 若存在 topic entity 没有任何有效的候选 path, 则略过该样本, 返回空信息
        2. 尝试将从不同 topic entity 出发的 paths 合并
        3. 从 path merging 的结果中导出 rel-rel rr_aps
    """
    ent_paths = dict()
    for ent in topic_ents:
        ent_paths[ent] = []
    # filter out invalid path
    for path_info in paths:
        ent, path = __parse_path_info(path_info)
        ent_paths[ent].append(path)
    # print(ent_paths)
    # skip case with incomplete path info
    for ent in ent_paths:
        if len(ent_paths[ent]) == 0:
            return dict()
    # merge paths && generate trips
    all_trips = []
    path_groups = __generate_path_groups(ent_paths)
    var_idx = 0
    for path_group in path_groups:
        var_idx, trips = __generate_trips_from_path_group(path_group, var_idx)
        all_trips += trips
    # print(path_groups)
    # obtain pp rr_aps from trips
    _, rel_rels = AdjacentInfoPrepare.parse_adjacent_info_from_trips(all_trips, False)
    return rel_rels
    
    
def __generate_trips_from_path_group(path_group: List[List[str]], var_idx: int) -> Tuple[int, List[List[str]]]:
    start_var = var_idx
    var_idx += 1
    ## step1: generate paths with var
    paths_with_var = []
    for path in path_group:
        temp = [start_var]
        for rel in reversed(path):
            temp.insert(0, rel)
            temp.insert(0, var_idx)
            var_idx += 1
        paths_with_var.append(temp)
    ## step2: generate paths with merged var
    paths_with_merged_var = deepcopy(paths_with_var)
    step = -1
    while True:
        changed = False
        # collect vars that can be merged
        merge_info = dict()
        for path in paths_with_merged_var:
            if len(path) + step < 2:
                continue
            cur_var = path[len(path) + step]
            cur_rel = path[len(path) + step - 1]
            next_var = path[len(path) + step - 2]
            key = str(cur_var) + cur_rel
            if key not in merge_info:
                merge_info[key] = set()
            merge_info[key].add(next_var)
        # merge vars
        merge_var_map = dict()
        for key in merge_info:
            if len(merge_info[key]) > 1:
                for var in merge_info[key]:
                    merge_var_map[var] = var_idx
                var_idx += 1
        for path in paths_with_merged_var:
            if len(path) + step < 2:
                continue
            if path[len(path) + step - 2] in merge_var_map:
                path[len(path) + step - 2] = merge_var_map[path[len(path) + step - 2]]
                changed = True
        if not changed:
            break
        step -= 2
    ## step3: generate trips based on paths with var
    facts = set()
    for path in paths_with_var + paths_with_merged_var:
        for idx in range(0, len(path) - 1, 2):
            facts.add(Fact([normalize_triplet([str(item) for item in path[idx:idx + 3]])]))
    trips = []
    for fact in facts:
        trips.append(fact.triplets[0])
    return var_idx, trips


def __generate_path_groups(ent_paths: Dict[str, List[List[str]]]) -> List[List[List[str]]]:
    ans = [[]]
    for ent in ent_paths:
        new_ans = []
        for path in ent_paths[ent]:
            for group in ans:
                temp = deepcopy(group)
                temp.append(path)
                new_ans.append(temp)
        ans = new_ans
    return ans


def __parse_path_info(path_info: List[str]) -> Tuple[str, List[str]]:
    assert path_info[0].startswith("ns:")
    ent = path_info[0]
    path = []
    for item in path_info[1:]:
        path += item.split("...")
    return ent, path


def __rr_aps_dict_to_str_list(rr_aps_dict) -> List[str]:
  ans = []
  for item1 in rr_aps_dict:
    for tag in rr_aps_dict[item1]:
      for item2 in rr_aps_dict[item1][tag]:
        ans.append(f"p1:{item1[3::]}, p2:{item2[3::]}, tag:{tag}")
  return ans


def analyze():
    if Config.ds_tag == 'WebQSP':
        return # in load ds items, only CWQ has valid logic form.
    cached_paths = read_jsonl_by_key(Config.cache_path)
    """ Step1: 通过验证集研究正例的结构导出率"""
    dev_items = load_ds_items(Config.ds_dev)
    total_cnt = 0
    path_hit = 0
    path_ap_info = {"num": 0, "S-S": 0, "S-O": 0, "O-S": 0, "O-O": 0}
    for idx, item in enumerate(tqdm(dev_items)):
        qid = item.id
        lf = item.lf
        topic_ents = item.topic_ents
        paths = cached_paths[qid]["paths"]
        _, gold_rr_aps = AdjacentInfoPrepare.parse_adjacent_info_from_trips(
            parse_triplets_from_serialized_ep(lf))
        path_rr_aps = get_rr_aps_from_searched_path(paths, topic_ents)
        if cover_gold_rr_aps(path_rr_aps, gold_rr_aps):
            path_hit += 1
        total_cnt += 1
        accumulate_rel_rel_statistics(path_rr_aps, path_ap_info)
    print(f">>> total:{total_cnt}, pHit:{path_hit}") # 
    print(f">>> path ap statistics:\n{path_ap_info}")
    print(1)
    # total:3519, pHit:2751, 78.18%
    # path ap statistics: {'num': 21057, 'S-S': 2169.331482169898, 'S-O': 2044.6838338080615, 'O-S': 2044.6838338080615, 'O-O': 1965.850228163022}


def td_gen(cached_paths, dest_file, ds_items, split):
    if os.path.exists(dest_file):
        print(f'{dest_file} already exists!')
        return
    if not os.path.exists(os.path.dirname(dest_file)):
        os.makedirs(os.path.dirname(dest_file))
    td = []
    for idx, item in enumerate(tqdm((ds_items))):
        qid = item.id
        topic_ents = item.topic_ents
        if qid not in cached_paths:
            paths = []
            print(f"{qid} not in paths.")
        else:
            paths = cached_paths[qid]["paths"]
        path_rel_rels = get_rr_aps_from_searched_path(paths, topic_ents)
        path_rr_aps = __rr_aps_dict_to_str_list(path_rel_rels)
        td.append({"split_id": f'{split}-{idx}', "id": qid, "question": item.question, "positive": path_rr_aps})
    write_json(td, dest_file)


if __name__ == '__main__':
    # analyze()
    """ 训练数据正例生成 """
    cached_paths = read_jsonl_by_key(Config.cache_path)
    # analyze()
    for split in ['dev', 'train']:
        dest_file = f"data/{Config.ds_tag}/ap_retrieval/training_data/{Config.ds_tag}_{split}_positive_rr_aps.json"
        ds_items = load_ds_items(Config.ds_split_f(split))
        td_gen(cached_paths, dest_file=dest_file, ds_items=ds_items, split=split)
