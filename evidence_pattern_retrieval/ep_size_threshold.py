# 通过训练集搜索出的路径确定组合 EP 时的 size threshold
# ds_tag = "CWQ"
# import sys
# sys.argv = ['config.py', '--config', f'config_{ds_tag}.yaml']

from config import Config
from my_utils.io_utils import read_jsonl_by_key
from my_utils.data_item import DataItem, load_ds_items

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
      ent_facts_cnt[path[0]] = min(ent_facts_cnt[path[0]], get_fact_cnt_4_path(path))
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