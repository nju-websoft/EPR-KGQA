import json

from my_utils.io_utils import read_json, append_line, read_jsonl_by_key, append_jsonl
import os
from my_utils.logger import Logger
from my_utils.pred_base import PredBase
from config import Config
from tqdm import tqdm

logger = Logger.get_logger("Trans_NSM",True)

def collect_covered_ent_rel(retrieved_subgs):
  ents = set()
  rels = set()
  for subg_info in tqdm(retrieved_subgs):
    for tripstr in subg_info["subg"]:
      trips = tripstr.split(" ")
      ents.add(trips[0][3::])
      ents.add(trips[2][3::])
      rels.add(trips[1][3::])
      if PredBase.get_reverse(trips[1]):
        rels.add(PredBase.get_reverse(trips[1])[3::])
    for topic in subg_info["topics"]:
      ents.add(topic[3::])
    for ans in subg_info["answers"]:
      ents.add(ans[3::])
  return ents,rels

def gen_ent_rel_list(train_subgs, dev_subgs, test_subgs, entity_file, relation_file):
  # if os.path.exists(entity_file) and os.path.exists(relation_file):
  #   logger.info("entity/relation list already collected.")
  #   return
  train_ents, train_rels = collect_covered_ent_rel(train_subgs)
  dev_ents, dev_rels = collect_covered_ent_rel(dev_subgs)
  test_ents, test_rels = collect_covered_ent_rel(test_subgs)
  all_ents = train_ents | dev_ents | test_ents
  all_rels = train_rels | dev_rels | test_rels
  logger.info(f"{len(all_ents)} unique entitied and {len(all_rels)} relations in top{Config.ep_topk} subgs")
  if os.path.exists(entity_file):
    os.remove(entity_file)
  if os.path.exists(relation_file):
    os.remove(relation_file)
  for ent in all_ents:
    append_line(ent, entity_file)
  for rel in sorted(all_rels):
    append_line(rel, relation_file)
    
def load_itemid_map(item_file):
  item2id = dict()
  if not os.path.exists(item_file):
    return item2id
  for line in open(item_file):
    item = line.strip()
    item2id[item] = len(item2id)
  return item2id
    
def trans_to_nsm_format(ent2id, rel2id, ours_subgs, reformated_subgs_f):
  reformated_subgs = []
  for subg_info in ours_subgs:
    qid = subg_info["id"]
    topic_ents = [ent2id[topic[3::]] for topic in subg_info["topics"]]
    subg_ents = []
    subg_tuples = []
    for tripstr in subg_info["subg"]:
      trips = tripstr.split(" ")
      subj = trips[0][3::]
      pred = trips[1][3::]
      if PredBase.get_reverse(trips[1]):
        pred_rev = PredBase.get_reverse(trips[1])[3::]
      else:
        pred_rev = None
      obj = trips[2][3::]
      subg_ents.append(ent2id[subj])
      subg_ents.append(ent2id[obj])
      subg_tuples.append([ent2id[subj], rel2id[pred], ent2id[obj]])
      if pred_rev:
        subg_tuples.append([ent2id[obj], rel2id[pred_rev], ent2id[subj]])
    subg_ents = list(set(subg_ents))
    nsm_item = dict()
    nsm_item["id"] = subg_info["id"]
    nsm_item["answers"] = [{"kb_id":ans[3::], "text":None} for ans in subg_info["answers"]]
    nsm_item["question"] = subg_info["question"]
    nsm_item["entities"] = topic_ents
    nsm_item["subgraph"] = dict()
    nsm_item["subgraph"]["entities"] = subg_ents
    nsm_item["subgraph"]["tuples"] = subg_tuples
    reformated_subgs.append(nsm_item)
  if os.path.exists(reformated_subgs_f):
    os.remove(reformated_subgs_f)
  for subg_info in reformated_subgs:
    append_jsonl(subg_info, reformated_subgs_f)


def trans_to_nsm_format_iid_setting(ent2id, rel2id, ours_subgs, reformated_subgs_f):
  with open(Config.ds_test_iid_idxs, 'r') as f:
    iid_idxs = set(json.load(f))
  reformated_subgs = []
  for idx, subg_info in enumerate(ours_subgs):
    if idx not in iid_idxs:
      continue
    qid = subg_info["id"]
    topic_ents = [ent2id[topic[3::]] for topic in subg_info["topics"]]
    subg_ents = []
    subg_tuples = []
    for tripstr in subg_info["subg"]:
      trips = tripstr.split(" ")
      subj = trips[0][3::]
      pred = trips[1][3::]
      if PredBase.get_reverse(trips[1]):
        pred_rev = PredBase.get_reverse(trips[1])[3::]
      else:
        pred_rev = None
      obj = trips[2][3::]
      subg_ents.append(ent2id[subj])
      subg_ents.append(ent2id[obj])
      subg_tuples.append([ent2id[subj], rel2id[pred], ent2id[obj]])
      if pred_rev:
        subg_tuples.append([ent2id[obj], rel2id[pred_rev], ent2id[subj]])
    subg_ents = list(set(subg_ents))
    nsm_item = dict()
    nsm_item["id"] = subg_info["id"]
    nsm_item["answers"] = [{"kb_id":ans[3::], "text":None} for ans in subg_info["answers"]]
    nsm_item["question"] = subg_info["question"]
    nsm_item["entities"] = topic_ents
    nsm_item["subgraph"] = dict()
    nsm_item["subgraph"]["entities"] = subg_ents
    nsm_item["subgraph"]["tuples"] = subg_tuples
    reformated_subgs.append(nsm_item)
  if os.path.exists(reformated_subgs_f):
    os.remove(reformated_subgs_f)
  for subg_info in reformated_subgs:
    append_jsonl(subg_info, reformated_subgs_f)


def trans_to_nsm_format_zero_shot_setting(ent2id, rel2id, ours_subgs, reformated_subgs_f):
  with open(Config.ds_test_zero_shot_idxs, 'r') as f:
    zero_shot_idxs = set(json.load(f))
  reformated_subgs = []
  for idx, subg_info in enumerate(ours_subgs):
    if idx not in zero_shot_idxs:
      continue
    qid = subg_info["id"]
    topic_ents = [ent2id[topic[3::]] for topic in subg_info["topics"]]
    subg_ents = []
    subg_tuples = []
    for tripstr in subg_info["subg"]:
      trips = tripstr.split(" ")
      subj = trips[0][3::]
      pred = trips[1][3::]
      if PredBase.get_reverse(trips[1]):
        pred_rev = PredBase.get_reverse(trips[1])[3::]
      else:
        pred_rev = None
      obj = trips[2][3::]
      subg_ents.append(ent2id[subj])
      subg_ents.append(ent2id[obj])
      subg_tuples.append([ent2id[subj], rel2id[pred], ent2id[obj]])
      if pred_rev:
        subg_tuples.append([ent2id[obj], rel2id[pred_rev], ent2id[subj]])
    subg_ents = list(set(subg_ents))
    nsm_item = dict()
    nsm_item["id"] = subg_info["id"]
    nsm_item["answers"] = [{"kb_id":ans[3::], "text":None} for ans in subg_info["answers"]]
    nsm_item["question"] = subg_info["question"]
    nsm_item["entities"] = topic_ents
    nsm_item["subgraph"] = dict()
    nsm_item["subgraph"]["entities"] = subg_ents
    nsm_item["subgraph"]["tuples"] = subg_tuples
    reformated_subgs.append(nsm_item)
  if os.path.exists(reformated_subgs_f):
    os.remove(reformated_subgs_f)
  for subg_info in reformated_subgs:
    append_jsonl(subg_info, reformated_subgs_f)


def prepare_nsm_input():
  nsm_ours_subg_f = lambda tag : f"data/dataset/{Config.ds_tag}_NSM/{tag}_simple.json"
  entity_file = f"data/dataset/{Config.ds_tag}_NSM/entities.txt"
  relation_file = f"data/dataset/{Config.ds_tag}_NSM/relations.txt"
  print(">>> read ours subg file")

  train_topk_subg = read_json(Config.ans_rank_td_f("train"))
  dev_topk_subg = read_json(Config.ans_rank_td_f("dev"))
  test_topk_subg = read_json(Config.induced_subg_f("test"))

  print(">>> collect entity/relation list")
  gen_ent_rel_list(train_topk_subg, dev_topk_subg, test_topk_subg, entity_file, relation_file)
  ent2id = load_itemid_map(entity_file)
  rel2id = load_itemid_map(relation_file)

  print(">>> reformat ours subgs")
  trans_to_nsm_format(ent2id, rel2id, train_topk_subg, nsm_ours_subg_f("train"))
  trans_to_nsm_format(ent2id, rel2id, dev_topk_subg, nsm_ours_subg_f("dev"))
  trans_to_nsm_format(ent2id, rel2id, test_topk_subg, nsm_ours_subg_f("test"))


def prepare_nsm_train_data_for_topk_study():
  nsm_ours_subg_f = lambda tag : f"data/dataset/{Config.ds_tag}_NSM/{tag}_simple.json"
  entity_file = f"data/dataset/{Config.ds_tag}_NSM/entities.txt"
  relation_file = f"data/dataset/{Config.ds_tag}_NSM/relations.txt"

  print(">>> collect used entities & relations")
  # collect entities & relations from induced subgraph (train lp=100, ep=3)
  Config.ep_topk = 3
  train_topk_subg = read_json(Config.ans_rank_td_f("train"))
  dev_topk_subg = read_json(Config.ans_rank_td_f("dev"))
  Config.ep_topk = 1
  test_topk_subg = read_json(Config.induced_subg_f("test"))

  train_ents, train_rels = collect_covered_ent_rel(train_topk_subg)
  dev_ents, dev_rels = collect_covered_ent_rel(dev_topk_subg)
  test_ents, test_rels = collect_covered_ent_rel(test_topk_subg)
  used_ents = train_ents | dev_ents | test_ents
  used_rels = train_rels | dev_rels | test_rels
  
  # collect entities & relations from induced subgraph (test lp=[20:200], ep=1)
  Config.ep_topk = 1
  for topk in range(20,201,20):
    Config.ap_topk = topk
    test_ents, test_rels = collect_covered_ent_rel(read_json(Config.induced_subg_f("test")))
    used_ents |= test_ents
    used_rels |= test_rels

  # write used entities & relations to file
  if os.path.exists(entity_file):
    os.remove(entity_file)
  if os.path.exists(relation_file):
    os.remove(relation_file)
  for ent in used_ents:
    append_line(ent, entity_file)
  for rel in sorted(used_rels):
    append_line(rel, relation_file)
    
  print(">>> reformat ours subgs")
  ent2id = load_itemid_map(entity_file)
  rel2id = load_itemid_map(relation_file)
  trans_to_nsm_format(ent2id, rel2id, train_topk_subg, nsm_ours_subg_f("train"))
  trans_to_nsm_format(ent2id, rel2id, dev_topk_subg, nsm_ours_subg_f("dev"))
  trans_to_nsm_format(ent2id, rel2id, test_topk_subg, nsm_ours_subg_f("test"))


def check_relations():
  relation_file = f"data/dataset/{Config.ds_tag}_NSM/relations.txt"
  rel2id = load_itemid_map(relation_file)

  print(">>> collect used relations")
  used_rels = set()
  # collect relations from induced subgraph (test lp=[20:200], ep=1)
  Config.ep_topk = 1
  for topk in range(20, 201, 20):
    Config.ap_topk = topk
    test_ents, test_rels = collect_covered_ent_rel(read_json(Config.induced_subg_f("test")))
    used_rels |= test_rels

  for rel in used_rels:
    if rel not in rel2id:
      print(rel)


def update_entity_and_relation_file_by_test_file():
  entity_file = f"data/dataset/{Config.ds_tag}_NSM/entities.txt"
  relation_file = f"data/dataset/{Config.ds_tag}_NSM/relations.txt"
  ent2id = load_itemid_map(entity_file)
  rel2id = load_itemid_map(relation_file)

  print(">>> collect used entities & relations")
  used_ents = set()
  used_rels = set()

  # collect entities & relations from induced subgraph (test lp=[20:200], ep=1)
  Config.ep_topk = 1
  for topk in range(100, 101, 10):
    Config.ap_topk = topk
    test_ents, test_rels = collect_covered_ent_rel(read_json(Config.induced_subg_f("test")))
    used_ents |= test_ents
    used_rels |= test_rels

  for ent in used_ents:
    if ent not in ent2id:
      append_line(ent, entity_file)
  for rel in used_rels:
    if rel not in rel2id:
      append_line(rel, relation_file)

  print(1)


def update_entity_and_relation_file_by_train_and_dev():
  nsm_ours_subg_f = lambda tag: f"data/dataset/{Config.ds_tag}_NSM/{tag}_simple.json"
  entity_file = f"data/dataset/{Config.ds_tag}_NSM/entities.txt"
  relation_file = f"data/dataset/{Config.ds_tag}_NSM/relations.txt"
  ent2id = load_itemid_map(entity_file)
  rel2id = load_itemid_map(relation_file)

  print(">>> collect used entities & relations")
  # collect entities & relations from induced subgraph (train lp=100, ep=3)
  Config.ap_topk = 100
  Config.ep_topk = 3
  train_topk_subg = read_json(Config.ans_rank_td_f("train"))
  dev_topk_subg = read_json(Config.ans_rank_td_f("dev"))

  train_ents, train_rels = collect_covered_ent_rel(train_topk_subg)
  dev_ents, dev_rels = collect_covered_ent_rel(dev_topk_subg)
  used_ents = train_ents | dev_ents
  used_rels = train_rels | dev_rels

  for ent in used_ents:
    if ent not in ent2id:
      append_line(ent, entity_file)
  for rel in used_rels:
    if rel not in rel2id:
      append_line(rel, relation_file)

  print(">>> reformat ours subgs")
  ent2id = load_itemid_map(entity_file)
  rel2id = load_itemid_map(relation_file)
  trans_to_nsm_format(ent2id, rel2id, train_topk_subg, nsm_ours_subg_f("train"))
  trans_to_nsm_format(ent2id, rel2id, dev_topk_subg, nsm_ours_subg_f("dev"))


def generate_test_file():
  entity_file = f"data/dataset/{Config.ds_tag}_NSM/entities.txt"
  relation_file = f"data/dataset/{Config.ds_tag}_NSM/relations.txt"
  ent2id = load_itemid_map(entity_file)
  rel2id = load_itemid_map(relation_file)
  print(Config.ap_topk)
  Config.ap_topk = 100
  Config.ep_topk = 1
  test_topk_subg = read_json(Config.induced_subg_f("test"))
  ours_nsm_subg_file = lambda tag: f"/home/jjyu/IRQA/data/dataset/{Config.ds_tag}_NSM/test_simple.json"
  trans_to_nsm_format(ent2id, rel2id, test_topk_subg, ours_nsm_subg_file("test"))


def generate_dev_test_file():
  entity_file = f"data/dataset/{Config.ds_tag}_NSM/entities.txt"
  relation_file = f"data/dataset/{Config.ds_tag}_NSM/relations.txt"
  ent2id = load_itemid_map(entity_file)
  rel2id = load_itemid_map(relation_file)
  Config.ap_topk = 100
  print(Config.ap_topk)
  test_topk_subg = read_json(Config.induced_subg_f("dev"))
  ours_nsm_subg_file = lambda tag: f"/home/jjyu/IRQA/data/dataset/{Config.ds_tag}_NSM/test_simple.json"
  trans_to_nsm_format(ent2id, rel2id, test_topk_subg, ours_nsm_subg_file("test"))


if __name__ == "__main__":
  # prepare_nsm_input()
  # prepare_nsm_train_data_for_topk_study()
  # check_relations()
  update_entity_and_relation_file_by_test_file()
  # update_entity_and_relation_file_by_train_and_dev()
  generate_test_file()
