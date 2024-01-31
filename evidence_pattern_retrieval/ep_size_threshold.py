# Self-defined Modules
from config import Config
from my_utils.io_utils import read_jsonl_by_key
from my_utils.data_item import load_ds_items


def get_fact_cnt_4_path(path):
    ans = 0
    for rel in path[1::]:
        if rel.find("...") != -1:
            ans += 2
        else:
            ans += 1
    return ans


def get_fact_cnt_4_item(paths):
    ans = 0
    ent_facts_cnt = dict()
    for path in paths:
        if path[0] not in ent_facts_cnt:
            ent_facts_cnt[path[0]] = get_fact_cnt_4_path(path)
        else:
            ent_facts_cnt[path[0]] = min(
                ent_facts_cnt[path[0]], get_fact_cnt_4_path(path)
            )
    for ent in ent_facts_cnt:
        ans += ent_facts_cnt[ent]
    return ans


train_items = load_ds_items(Config.ds_train)
cached_paths = read_jsonl_by_key(Config.cache_path)

paths_fact_cnt = dict()
for item in train_items:
    if item.id not in cached_paths:
        continue
    paths = cached_paths[item.id]["paths"]
    fact_cnt = get_fact_cnt_4_item(paths)
    if fact_cnt not in paths_fact_cnt:
        paths_fact_cnt[fact_cnt] = 0
    paths_fact_cnt[fact_cnt] += 1
print(paths_fact_cnt)
# CWQ {2: 11287, 3: 7433, 4: 2708, 0: 1473, 5: 947, 1: 3165, 6: 562, 7: 48, 8: 8}
# WebQSP {2: 865, 1: 1835, 4: 30, 0: 88, 6: 5, 3: 3}
# the thresholds for CWQ and WebQSP are set to 5 and 3 respectively.