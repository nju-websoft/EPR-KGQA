from typing import *
from my_utils.freebase import FreebaseODBC
from my_utils.ap_utils import parse_triplets_from_serialized_cg


def is_instantiable(cg:str) -> bool:
  # query = f""" SPARQL
  #   SELECT ?j
  #   WHERE {{{cg}}}
  #   LIMIT 1
  # """
  query = f""" SPARQL
    SELECT ?j
    WHERE {{{cg}}}
    LIMIT 1000
  """
  return FreebaseODBC.has_query_results(query)


def get_instantiable_top1(sorted_cg:List[str], jump_check:bool) -> str:
  ans = None
  for cg in sorted_cg:
    if jump_check:
      ans = cg
      break
    if is_instantiable(cg):
      ans = cg
      # print(cg)
      break
  return ans


def get_instantiable_topk(sorted_cg:List[str], topk:int) -> List[str]:
  ans = []
  for cg in sorted_cg:
    if is_instantiable(cg):
      ans.append(cg)
      if len(ans) >= topk:
        break
  return ans


def get_cg_instance_var_map(cg:str) -> Dict[str, List[str]]:
  vars = set()
  trips = parse_triplets_from_serialized_cg(cg)
  for trip in trips:
    if trip[0].startswith("?"):
      vars.add(trip[0])
    if trip[2].startswith("?"):
      vars.add(trip[2])
  query = f""" SPARQL
    SELECT {" ".join(vars)}
    WHERE {{{cg}}}
    LIMIT 1000
  """
  res = FreebaseODBC.get_query_results(query, vars)
  return res


def __get_row_num_from_var_map(var_map:Dict[str, List[str]]) -> int:
  for var in var_map:
    return len(var_map[var])
  return 0


def induce_instantiated_subg(cg_str:str) -> Tuple[List[str], List[str]]:
  """ return pair, pair[0]:trips, pair[1]:nodes """
  trip_strs = set()
  nodes = set()
  var_const_map = get_cg_instance_var_map(cg_str)
  trips_with_var = parse_triplets_from_serialized_cg(cg_str)
  row_cnt = __get_row_num_from_var_map(var_const_map)
  for trip in trips_with_var:
    subj = trip[0]
    pred = trip[1]
    obj = trip[2]
    if subj.startswith("?"):
      subj_insts = var_const_map[subj]
    else:
      subj_insts = [subj for i in range(row_cnt)]
    if obj.startswith("?"):
      obj_insts = var_const_map[obj]
    else:
      obj_insts = [obj for i in range(row_cnt)]
    for idx in range(row_cnt):
      s = subj_insts[idx]
      o = obj_insts[idx]
      # 过滤 s 和 o 不是有效实体的情况
      if s.startswith("ns:") and o.startswith("ns:"):
        trip_strs.add(" ".join([s,pred,o]))
        nodes.add(s)
        nodes.add(o)
  return list(trip_strs), list(nodes)


def integrate_topk_instantiated_subg(cg_strs:List[str], topk) -> Tuple[List[str],Dict[str,str],List[str]]:
  """ return pair, pair[0]:trips, pair[1]:node_cg_map """
  trip_strs = set()
  node_cgs_map = dict()
  valid_cg_cnt = 0
  valid_cgs = []
  for idx, cg in enumerate(cg_strs):
    if valid_cg_cnt >= topk:
      break
    trips, nodes = induce_instantiated_subg(cg)
    if len(trips) == 0:
      continue
    else:
      valid_cg_cnt += 1
      valid_cgs.append(cg)
    trip_strs |= set(trips)
    for node in nodes:
      if node not in node_cgs_map:
        node_cgs_map[node] = str(valid_cg_cnt-1)
      else:
        node_cgs_map[node] += f" {valid_cg_cnt-1}"
  return list(trip_strs), node_cgs_map, valid_cgs


def get_retrieved_lps(topic_ents:List[str], lp_info:dict) -> Tuple[dict, dict]:
  """ prepare two types of lps based on lp retrieval results 
      [param] lp_info: {"id":str, "split_id":str, "question":str, "candidate_snippets":List[str]}
      e.g. government.politician.government_positions_held S-S government.government_position_held.jurisdiction_of_office
  """
  ent_preds = dict()
  pred_preds = dict()
  # prepare ent_pred
  # 如果文件给出了 ent_preds 结果，则直接读取。否则需要临时查询
  if "ent_preds" in lp_info:
    for ep_str in lp_info["ent_preds"]:
      temp = ep_str.split(" ")
      ent = temp[0]
      tag = temp[1]
      pred = temp[2]
      if ent not in ent_preds:
        ent_preds[ent] = {"fwd":[], "rev":[]}
      ent_preds[ent][tag].append(pred)
      if pred not in pred_preds:
        pred_preds[pred] = {"S-S":[], "S-O":[], "O-S":[], "O-O":[]}
  else:
    for te in topic_ents:
      ent_preds[te] = {"fwd":[], "rev":[]}
      neighbor_preds = FreebaseODBC.query_neighbor_preds([te])
      for pred in neighbor_preds:
        if pred.endswith("_Rev"):
          pred = pred[:-4]
          ent_preds[te]["rev"].append(pred)
        else:
          ent_preds[te]["fwd"].append(pred)
        if pred not in pred_preds:
          pred_preds[pred] = {"S-S":[], "S-O":[], "O-S":[], "O-O":[]}
  # prepare pred_pred
  for pp_str in lp_info["candidates"]:
    splits = pp_str.split(" ")
    pred1 = "ns:"+splits[0]
    pred2 = "ns:"+splits[2]
    tag = splits[1]
    if pred1 not in pred_preds:
      pred_preds[pred1] = {"S-S":[], "S-O":[], "O-S":[], "O-O":[]}
    pred_preds[pred1][tag].append(pred2)
  return ent_preds, pred_preds