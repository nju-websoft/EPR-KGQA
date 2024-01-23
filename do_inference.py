from typing import *
from tqdm import tqdm
import time
import argparse

from my_utils.data_item import DataItem, load_ds_items
from config import Config
from my_utils.io_utils import *
from my_utils.ap_utils import filter_topk_lps, parse_snippet_dict_from_strs
from evidence_pattern_retrieval.ep_construction import EPCombiner
from my_utils.ep_utils import *


def snippet_dict_to_str_list(snippet_dict) -> List[str]:
  ans = []
  for item1 in snippet_dict:
    for tag in snippet_dict[item1]:
      for item2 in snippet_dict[item1][tag]:
        ans.append(f"{item1[3::]} {tag} {item2[3::]}")
  return ans


def generate_candidate_eps(data_items:List[DataItem], ranked_lp_info:List[dict], lp_topk:int) -> List[dict]:
  print(f"[Generate Candidate Eps] {len(data_items)} items total, lp_topk: {lp_topk}")
  ans = []
  retrieved_lps = filter_topk_lps(ranked_lp_info, lp_topk)
  assert len(data_items) == len(retrieved_lps)
  idx = -1
  for item, lp_info in tqdm(zip(data_items, retrieved_lps)):
    assert item.id == lp_info["id"]
    ep_lps, pp_lps = get_retrieved_lps(item.topic_ents, lp_info)
    if len(ep_lps) == 0:
      combs = []
    else:
      combs = EPCombiner.combine(ep_lps, pp_lps, False)
    ans.append({"id":item.id, "question":item.question, "candidates":[comb.get_query_trips() for comb in combs]})
  return ans
    
    
def induce_subg_from_topk_cgs_in_batch(data_items:List[DataItem], ranked_cg_info:List[dict], topk=Config.ep_topk) -> List[dict]:
  print(f"[Induce Subgraph] {len(data_items)} items total, lp_topk: {Config.ap_topk}, ep_topk: {Config.ep_topk}")
  ranked_cgs = []
  for info in ranked_cg_info:
    ranked_cgs.append([pair[0] for pair in info["sorted_candidates_with_logits"]])
  ans = []
  assert len(data_items) == len(ranked_cgs)
  for item, ranked_cg in tqdm(zip(data_items, ranked_cgs)):
    qid = item.id
    if qid != 'WebQTest-59':
      continue
    question = item.question
    trips, node_cg_map, topk_cgs = integrate_topk_instantiated_subg(ranked_cg, topk)
    ans.append({"id":qid, "question":question, "answers":item.answers, "topics":item.topic_ents, "topk_cg":topk_cgs, "subg":trips, "node_cgs":node_cg_map})
  analyze_subg_info(data_items, ans)
  return ans
    
    
def calculate_eval_metric_in_batch(data_items:List[DataItem], predict_res_f:str) -> Tuple[float, float]:
  predict_res = read_json(predict_res_f)
  macro_hits1 = 0
  macro_f1 = 0
  for idx, item in tqdm(enumerate(data_items)):
    gold_ans = item.answers
    pred_res = ["ns:"+mid for mid in predict_res[idx]]
    _, _, f1 = calculate_PRF1(gold_ans, pred_res)
    _, _, f1_check, _ = evaluation_by_question(gold_ans, pred_res)
    hits1 = calculate_hits1(gold_ans, pred_res)
    macro_f1 += f1
    macro_hits1 += hits1
    if f1 != f1_check:
      print(f"gold:{gold_ans}")
      print(f"predict:{predict_res[idx]}")
      print(f"f1:{f1}")
      print(f"f1_check:{f1_check}")
  return macro_hits1/len(data_items), macro_f1/len(data_items)


def calculate_eval_metric_of_ranked_ans(data_items:List[DataItem], ranked_ans_f:str) -> Tuple[float, float]:
  ranked_ans = read_json(ranked_ans_f)
  macro_hits1 = 0
  macro_f1 = 0
  for ds_item, rank_info in zip(data_items, ranked_ans):
    assert ds_item.id == rank_info["ID"]
    gold_ans = ds_item.answers
    predict_ans = []
    for ans_info in rank_info["sorted_candidates_with_logits"]:
      if len(predict_ans) == 0:
        predict_ans.append(ans_info[0]["mid"])
      elif ans_info[1] > 0:
        predict_ans.append(ans_info[0]["mid"])
    _, _, f1 = calculate_PRF1(gold_ans, predict_ans)
    hits1 = calculate_hits1(gold_ans, predict_ans, False)
    macro_f1 += f1
    macro_hits1 += hits1
  return macro_hits1/len(data_items), macro_f1/len(data_items)
    

def analyze_subg_info(data_items:List[DataItem], instantiated_subgs:List[dict]):
  total = 0
  node_cnt = 0
  trip_cnt = 0
  ans_hit = 0
  for item, info in zip(data_items, instantiated_subgs):
    total += 1
    node_cnt += len(info["node_cgs"])
    trip_cnt += len(info["subg"])
    if len(set(item.answers) & set(info["node_cgs"].keys())) > 0:
      ans_hit += 1
  ans_hit = ans_hit/total
  node_avg = node_cnt/total
  trip_avg = trip_cnt/total
  print(f"total:{total}, avg #nodes:{node_avg}, avg #trips:{trip_avg}, ans hits:{ans_hit}")
  

def analyze_subg_info_iid_setting(data_items:List[DataItem], instantiated_subgs:List[dict]):
  total = 0
  node_cnt = 0
  trip_cnt = 0
  ans_hit = 0
  with open(Config.ds_test_iid_idxs, 'r') as f:
    iid_idxs = set(json.load(f))
  idx = -1
  for item, info in zip(data_items, instantiated_subgs):
    idx += 1
    if idx not in iid_idxs:
      continue
    total += 1
    node_cnt += len(info["node_cgs"])
    trip_cnt += len(info["subg"])
    if len(set(item.answers) & set(info["node_cgs"].keys())) > 0:
      ans_hit += 1
  ans_hit = ans_hit/total
  node_avg = node_cnt/total
  trip_avg = trip_cnt/total
  print(f"total:{total}, avg #nodes:{node_avg}, avg #trips:{trip_avg}, ans hits:{ans_hit}")


def analyze_subg_info_zero_shot_setting(data_items:List[DataItem], instantiated_subgs:List[dict]):
  total = 0
  node_cnt = 0
  trip_cnt = 0
  ans_hit = 0
  with open(Config.ds_test_zero_shot_idxs, 'r') as f:
    zero_shot_idxs = set(json.load(f))
  idx = -1
  for item, info in zip(data_items, instantiated_subgs):
    idx += 1
    if idx not in zero_shot_idxs:
      continue
    total += 1
    node_cnt += len(info["node_cgs"])
    trip_cnt += len(info["subg"])
    if len(set(item.answers) & set(info["node_cgs"].keys())) > 0:
      ans_hit += 1
  ans_hit = ans_hit/total
  node_avg = node_cnt/total
  trip_avg = trip_cnt/total
  print(f"total:{total}, avg #nodes:{node_avg}, avg #trips:{trip_avg}, ans hits:{ans_hit}")


def generate_candidate_eps_with_time_info(items:List[DataItem], split_tag:str, topk_range:int=200, step:int=20):
  topk_time_info = dict()
  topk_time_wo_io = dict()
  raw_topk = Config.ap_topk
  topks = range(topk_range[0], topk_range[1]+1, step)
  for topk in topks:
    Config.ap_topk = topk
    start = time.time()
    ranked_lp_info = read_json(Config.retrieved_ap_f(split_tag))
    start_wo_io = time.time()
    candi_eps = generate_candidate_eps(items, ranked_lp_info, Config.ap_topk)
    end_wo_io = time.time()
    write_json(candi_eps, Config.candi_ep_f(split_tag))
    end = time.time()
    topk_time_info[f"top{topk}"] = round(end - start, 1)
    topk_time_wo_io[f"top{topk}"] = round(end_wo_io - start_wo_io, 1)
  Config.ap_topk = raw_topk
  # print(f"Time Costs:{topk_time_info}")
  print(f"Time Costs (w/o IO):{topk_time_wo_io}")
  
  
def induce_subgraph_with_time_info(items:List[DataItem], split_tag:str, topk_range:List[int]=[20,200], step:int=20):
  topk_time_info = dict()
  topk_time_wo_io = dict()
  raw_topk = Config.ap_topk
  topks = range(topk_range[0], topk_range[1] + 1, step)
  for topk in topks:
    Config.ap_topk = topk
    start = time.time()
    ranked_ep_info = read_json(Config.ranked_ep_f(split_tag))
    start_wo_io = time.time()
    induced_subg = induce_subg_from_topk_cgs_in_batch(items, ranked_ep_info, topk=Config.ep_topk)
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
  print('')
  for topk in topks:
    Config.ap_topk = topk
    subgs = read_json(Config.induced_subg_f(split_tag))
    print(topk, end=": ")
    analyze_subg_info_iid_setting(data_items, subgs)
  print('')
  for topk in topks:
    Config.ap_topk = topk
    subgs = read_json(Config.induced_subg_f(split_tag))
    print(topk, end=": ")
    analyze_subg_info_zero_shot_setting(data_items, subgs)
  Config.ap_topk = raw_topk

def run(split_filename:str, split_tag:str, topk_range:List[int]=[20,200], step:int=20):
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