from typing import *
import re

from tqdm import tqdm
from my_utils.pred_base import PredBase

def filter_topk_preds(ranked_pred_list:List[str], topk:int, deduplicate:bool) -> List[str]:
  # 预处理，确保 predicate 都有前缀
  for idx, pred in enumerate(ranked_pred_list):
    if not pred.startswith("ns:"):
      ranked_pred_list[idx] = "ns:" + pred
  temp_list = []
  temp_set = set()
  if deduplicate:
    for pred in ranked_pred_list:
      if PredBase.get_reverse(pred) in temp_set:
        continue
      temp_list.append(pred)
      temp_set.add(pred)
  else:
    temp_list = ranked_pred_list
  return temp_list[:topk]

def get_snippet_str_set(snippets_info:dict) -> Set[str]:
  ans = set()
  for item1 in snippets_info:
    for tag in snippets_info[item1]:
      for item2 in snippets_info[item1][tag]:
        ans.add(" ".join([item1, item2, tag]))
  return ans

def expand_ent_snippet_str_set(ent:str, pred:str, tag:str) -> Set[str]:
  trans_dict = {"rev":"fwd", "fwd":"rev"}
  ans = set()
  ans.add(" ".join([ent, pred, tag]))
  rev = PredBase.get_reverse(pred)
  if rev != None:
    new_tag = trans_dict[tag]
    ans.add(" ".join([ent, rev, new_tag]))
  return ans

def expand_pred_snippet_str_set(pred1:str, pred2:str, tag:str) -> Set[str]:
  trans_dict = {"S":"O", "O":"S"}
  ans = set()
  ans.add(" ".join([pred1, pred2, tag]))
  rev1 = PredBase.get_reverse(pred1)
  rev2 = PredBase.get_reverse(pred2)
  if rev2 != None:
    ans.add(" ".join([pred1,rev2,tag[0]+"-"+trans_dict[tag[2]]]))
  if rev1 != None:
    ans.add(" ".join([rev1,pred2,trans_dict[tag[0]]+"-"+tag[2]]))
  if rev1 != None and rev2 != None:
    ans.add(" ".join([rev1,rev2,trans_dict[tag[0]]+"-"+trans_dict[tag[2]]]))
  return ans

def parse_triplets_from_serialized_cg(sparql:str) -> List[List[str]]:
  # TODO: 之后需要删掉这个函数（开源）
  if sparql.find("MANUAL SPARQL")!=-1:
    return []
  tripstrs = re.findall(r"[\?n][^ ()?{}]+ ns:[^ ]+ [\?n][^ ()?{}]+",sparql)
  # 处理三元组以.结尾的特殊情况
  for idx, s in enumerate(tripstrs):
    if s.endswith("."):
      tripstrs[idx] = tripstrs[idx][:-1]
  trips = []
  for tripstr in tripstrs:
    trip = tripstr.split(" ")
    assert len(trip) == 3
    obj = trip[2]
    if obj.startswith("?sk") or obj.startswith("?num"):
      continue
    else:
      trips.append(trip)
  return trips

def accumulate_pred_pred_statistics(pred_preds:Dict[str, Dict[str, List[str]]], statistics:Dict[str,float]):
  if len(pred_preds) == 0:
    return
  tags = ["S-S","S-O","O-O","O-S"]
  temp = {"num":0, "S-S":0, "S-O":0, "O-O":0, "O-S":0}
  for pred in pred_preds:
    temp["num"] += 1
    for tag in tags:
      temp[tag] += len(pred_preds[pred][tag])
  statistics["num"] += temp["num"]
  for tag in tags:
    statistics[tag] += temp[tag]/temp["num"]
    
def normalize_triplet(triplet:List[str]) -> List[str]:
    if triplet[1].endswith("_Rev"):
      return [triplet[2], triplet[1][:-4], triplet[0]]
    else:
      return triplet
    
def parse_pp_info_from_ranked_snippets(ranked_snippets:List[str], ranked_preds:List[str]) -> Dict[str, Dict[str, List[str]]]:
  tag_map = {"S-S":"S-S", "S-O":"O-S", "O-S":"S-O", "O-O":"O-O"}
  ans = dict()
  for pred in ranked_preds:
    ans[pred] = {"S-S":set(), "S-O":set(), "O-S":set(), "O-O":set()}
  for snp in ranked_snippets:
    temp = snp.split(" ")
    p1 = "ns:"+temp[0]
    tag = temp[1]
    p2 = "ns:"+temp[2]
    ans[p1][tag].add(p2)
    ans[p2][tag_map[tag]].add(p1)
  return ans

def parse_snippet_dict_from_strs(topic_ents:List[str], ent_pred_strs:List[str], pred_pred_strs:List[str]) -> Tuple[dict,dict]:
  pp_tag_map = {"S-S":"S-S", "S-O":"O-S", "O-S":"S-O", "O-O":"O-O"}
  ent_preds = dict()
  pred_preds = dict()
  for ent in topic_ents:
    ent_preds[ent] = {"fwd":set(), "rev":set()}
  
  for ep in ent_pred_strs:
    temp = ep.split(" ")
    ent = "ns:"+temp[0]
    tag = temp[1]
    pred = "ns:"+temp[2]
    if ent not in ent_preds:
      ent_preds[ent] = {"fwd":set(), "rev":set()}
    ent_preds[ent][tag].add(pred)
    # 补充 pp 中的信息
    if pred not in pred_preds:
      pred_preds[pred] = {"S-S":set(), "S-O":set(), "O-S":set(), "O-O":set()}
      
  for pp in pred_pred_strs:
    temp = pp.split(" ")
    p1 = "ns:"+temp[0]
    tag = temp[1]
    p2 = "ns:"+temp[2]
    if p1 not in pred_preds:
      pred_preds[p1] = {"S-S":set(), "S-O":set(), "O-S":set(), "O-O":set()}
    if p2 not in pred_preds:
      pred_preds[p2] = {"S-S":set(), "S-O":set(), "O-S":set(), "O-O":set()}
    pred_preds[p1][tag].add(p2)
    pred_preds[p2][pp_tag_map[tag]].add(p1)
  
  return ent_preds, pred_preds


def filter_topk_lps(ranked_lps:List[dict], topk:int) -> List[dict]:
  """ merge equivalent lps && filter topk && expand lp by tag
      e.g. [(p1,p2,S-S), (p1_rev,p3,S-O), (p2_rev,p3_rev,O-O)] -> [(p1,p2,S-S), (p1,p3,O-O), (P2,P3,S-S)]
      [param] ranked_lps: [{id:str, split_id:str, question:str, candidates:[str,...]}...]
  """
  trans_dict = {"S":"O", "O":"S"}
  for item in tqdm(ranked_lps):
    used_preds = set()
    topk_lps = set()
    expanded_lps = set()
    # merge reversed-equivalent lps && filter topk
    for lp in item["candidates"]:
      if len(topk_lps) >= topk:
        break
      temp = lp.split(" ")
      pred1 = temp[0]
      tag = temp[1]
      pred2 = temp[2]
      rev1 = PredBase.get_reverse("ns:"+pred1)
      rev2 = PredBase.get_reverse("ns:"+pred2)
      if rev1 != None and rev1[3::] in used_preds:
        pred1 = rev1[3::]
        tag = trans_dict[tag[0]] + tag[1] + tag[2]
      used_preds.add(pred1)
      if rev2 != None and rev2[3::] in used_preds:
        pred2 = rev2[3::]
        tag = tag[0] + tag[1] + trans_dict[tag[2]]
      used_preds.add(pred2)
      topk_lps.add(" ".join([pred1,tag,pred2]))
      # expand lp by tag
      new_tag = tag[2]+tag[1]+tag[0]
      expanded_lps.add(" ".join([pred2,new_tag,pred1]))
    item["candidates"] = list(topk_lps | expanded_lps)
  return ranked_lps