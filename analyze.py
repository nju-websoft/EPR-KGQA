from my_utils.data_item import DataItem, load_ds_items
from my_utils.io_utils import *
from config import Config
from typing import *
import pandas as pd
from my_utils.ep_utils import get_instantiable_topk
from tqdm import tqdm


def get_item_feature(item:DataItem) -> Set[str]:
  topic_ent_cnt = len(set(item.topic_ents))
  answer_ent_cnt = len(set(item.answers))
  comp_type = item.comp_type
  ans = set()
  if topic_ent_cnt == 1:
    ans.add("single_topic")
  elif topic_ent_cnt > 1:
    ans.add("multi_topic")
  if answer_ent_cnt == 1:
    ans.add("single_answer")
  elif answer_ent_cnt > 1:
    ans.add("multi_answer")
  # ans.add(f"#topic:{topic_ent_cnt}")
  # ans.add(f"#ans:{answer_ent_cnt}")
  ans.add(f"{comp_type}")
  return ans


def collect_nsm_test_p_r_f_hit(test_items:List[DataItem], nsm_test_res_f:str) -> dict:
  nsm_res = read_json(nsm_test_res_f)
  ans = dict()
  for idx, res in enumerate(nsm_res):
    qid = test_items[idx].id
    p = res["P"]
    r = res["R"]
    f1 = res["f1"]
    hits1 = res["hit"]
    ans[qid] = {"P":p, "R":r, "F1":f1, "Hits@1":hits1}
  return ans


def collect_ours_test_p_r_f_hit(test_items:List[DataItem], ours_test_res_f:str) -> dict:
  ours_res = read_json(ours_test_res_f)
  ans = dict()
  for idx, res in enumerate(ours_res):
    qid = test_items[idx].id
    answers = test_items[idx].answers
    predict_ans = []
    for ans_info in ours_res[idx]["sorted_candidates_with_logits"]:
      if len(predict_ans) == 0:
        predict_ans.append(ans_info[0]["mid"])
      elif ans_info[1] > 0:
        predict_ans.append(ans_info[0]["mid"])
    p, r, f1 = calculate_PRF1(answers, predict_ans)
    hits1 = calculate_hits1(answers, predict_ans, False)
    ans[qid] = {"P":p, "R":r, "F1":f1, "Hits@1":hits1}
  return ans


def accumulate_statistics(statistics_info:dict, metric_info:dict):
  statistics_info["count"] += 1
  for metric in metric_info:
    statistics_info[metric] += metric_info[metric]


def normalize_statistics(statistics_info:Dict[str,dict]):
  for feature in statistics_info:
    metrics = statistics_info[feature]
    metrics["P"] = round(metrics["P"]/metrics["count"],4)
    metrics["R"] = round(metrics["R"]/metrics["count"],4)
    metrics["F1"] = round(metrics["F1"]/metrics["count"],4)
    metrics["Hits@1"] = round(metrics["Hits@1"]/metrics["count"],4)


def calculate_metrics_by_feature(test_items:List[DataItem], test_res:dict) -> Dict[str,Dict[str,float]]:
  ans = dict()
  for item in test_items:
    qid = item.id
    features = get_item_feature(item)
    features.add("total")
    metric_res = test_res[qid]
    for ft in features:
      if ft not in ans:
        ans[ft] = {"count":0, "P":0.0, "R":0.0, "F1":0.0, "Hits@1":0.0}
      accumulate_statistics(ans[ft], metric_res)
  normalize_statistics(ans)
  return ans


def print_statistics_with_pd_format(statistics_info:Dict[str,dict]):
  target = []
  feature_tags = ["single_topic", "multi_topic", "single_answer", "multi_answer",
                 "conjunction", "composition", "superlative", "comparative", "total"]
  for feature in feature_tags:
    temp = {"feature":feature}
    for metric in statistics_info[feature]:
      temp[metric] = statistics_info[feature][metric]
    target.append(temp)
  df = pd.DataFrame(target)
  print(df)


def collect_top10_cg(ranked_cg_f:str, top10_cg_inst_f:str):
  ranked_cg_info = read_json(ranked_cg_f)
  ans = []
  for idx, item in tqdm(enumerate(ranked_cg_info)):
    qid = item["ID"]
    question = item["question"]
    ranked_cg = [info[0] for info in item["sorted_candidates_with_logits"]]
    top10_cgs = get_instantiable_topk(ranked_cg, 10)
    ans.append({"id":qid, "text":question, "top10_cgs":top10_cgs})
  write_json(ans, top10_cg_inst_f)


# def collect_top10_


test_items = load_ds_items(Config.ds_test)
nsm_test_res = collect_nsm_test_p_r_f_hit(test_items, "data/NSM/CWQTestResult.json")
ours_test_res = collect_ours_test_p_r_f_hit(test_items, Config.ranked_ans_f("test"))

print(">>> NSM-h test metrics:")
print_statistics_with_pd_format(calculate_metrics_by_feature(test_items, nsm_test_res))
print(">>> ours test metrics:")
print_statistics_with_pd_format(calculate_metrics_by_feature(test_items, ours_test_res))

# print(">>> topk cg node analysis:")
# # collect_top10_cg(Config.ranked_cg_f("dev"), "data/temp/cg_ranking/dev_top10_instantiable_cg.json")


# calculate answer hits and size of topk-cg instantiated subgraph
# cg_topk = 10
# test_items = load_ds_items(Config.ds_test)
# test_topk_cg_subg = json.load(open(f"data/temp/ans_ranking/CWQ_test_top20_preds_top100_pps_top{cg_topk}_cgs_instantiated_subg.json"))
# topk_hits = [0 for i in range(cg_topk)]
# topk_size = [0 for i in range(cg_topk)]
# for ds_item, subg_info in tqdm(zip(test_items, test_topk_cg_subg)):
#   answer_set = set(ds_item.answers)
#   topk_nodes = [set() for i in range(cg_topk)]
#   for node in subg_info["node_cgs"]:
#     rank1_idx = int(subg_info["node_cgs"][node].split(" ")[0])
#     for idx in range(rank1_idx, cg_topk):
#       topk_nodes[idx].add(node)
#   topk_trips = [[] for i in range(cg_topk)]
#   for trip_str in subg_info["subg"]:
#     trip = trip_str.split(" ")
#     for idx in range(cg_topk):
#       if trip[0] in topk_nodes[idx] and trip[2] in topk_nodes[idx]:
#         topk_trips[idx].append(trip_str)
#   for idx in range(cg_topk):
#     nodes = topk_nodes[idx]
#     trips = topk_trips[idx]
#     if len(nodes & answer_set) != 0:
#       topk_hits[idx] += 1
#     topk_size[idx] += len(trips)
# # normalize
# for idx in range(cg_topk):
#   topk_hits[idx] = round(topk_hits[idx] / len(test_items), 4)
#   topk_size[idx] = round(topk_size[idx] / len(test_items), 4)

# print(f"Topk_hits:{topk_hits}")
# print(f"Topk_size:{topk_size}")
