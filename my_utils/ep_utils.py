# Stardard Libraries
from typing import *

# Self-defined Modules
from my_utils.ap_utils import parse_triplets_from_serialized_ep
from my_utils.freebase import FreebaseODBC


def is_instantiable(ep: str) -> bool:
    # query = f""" SPARQL
    #   SELECT ?j
    #   WHERE {{{ep}}}
    #   LIMIT 1
    # """
    query = f""" SPARQL
    SELECT ?j
    WHERE {{{ep}}}
    LIMIT 1000
  """
    return FreebaseODBC.has_query_results(query)


def get_instantiable_top1(sorted_ep: List[str], jump_check: bool) -> str:
    ans = None
    for ep in sorted_ep:
        if jump_check:
            ans = ep
            break
        if is_instantiable(ep):
            ans = ep
            # print(ep)
            break
    return ans


def get_instantiable_topk(sorted_ep: List[str], topk: int) -> List[str]:
    ans = []
    for ep in sorted_ep:
        if is_instantiable(ep):
            ans.append(ep)
            if len(ans) >= topk:
                break
    return ans


def get_ep_instance_var_map(ep: str) -> Dict[str, List[str]]:
    vars = set()
    trips = parse_triplets_from_serialized_ep(ep)
    for trip in trips:
        if trip[0].startswith("?"):
            vars.add(trip[0])
        if trip[2].startswith("?"):
            vars.add(trip[2])
    query = f""" SPARQL
    SELECT {" ".join(vars)}
    WHERE {{{ep}}}
    LIMIT 1000
  """
    res = FreebaseODBC.get_query_results(query, vars)
    return res


def __get_row_num_from_var_map(var_map: Dict[str, List[str]]) -> int:
    for var in var_map:
        return len(var_map[var])
    return 0


def induce_instantiated_subg(ep_str: str) -> Tuple[List[str], List[str]]:
    """return pair, pair[0]:trips, pair[1]:nodes"""
    trip_strs = set()
    nodes = set()
    var_const_map = get_ep_instance_var_map(ep_str)
    trips_with_var = parse_triplets_from_serialized_ep(ep_str)
    row_cnt = __get_row_num_from_var_map(var_const_map)
    for trip in trips_with_var:
        subj = trip[0]
        rel = trip[1]
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
                trip_strs.add(" ".join([s, rel, o]))
                nodes.add(s)
                nodes.add(o)
    return list(trip_strs), list(nodes)


def integrate_topk_instantiated_subg(
    ep_strs: List[str], topk
) -> Tuple[List[str], Dict[str, str], List[str]]:
    """return pair, pair[0]:trips, pair[1]:node_ep_map"""
    trip_strs = set()
    node_eps_map = dict()
    valid_ep_cnt = 0
    valid_eps = []
    for idx, ep in enumerate(ep_strs):
        if valid_ep_cnt >= topk:
            break
        trips, nodes = induce_instantiated_subg(ep)
        if len(trips) == 0:
            continue
        else:
            valid_ep_cnt += 1
            valid_eps.append(ep)
        trip_strs |= set(trips)
        for node in nodes:
            if node not in node_eps_map:
                node_eps_map[node] = str(valid_ep_cnt - 1)
            else:
                node_eps_map[node] += f" {valid_ep_cnt-1}"
    return list(trip_strs), node_eps_map, valid_eps


def get_retrieved_aps(topic_ents: List[str], ap_info: dict) -> Tuple[dict, dict]:
    """prepare two types of aps based on ap retrieval results
    [param] ap_info: {"id":str, "split_id":str, "question":str, "candidate_snippets":List[str]}
    e.g. government.politician.government_positions_held S-S government.government_position_held.jurisdiction_of_office
    """
    ent_rels = dict()
    rel_rels = dict()
    # prepare ent_rel
    # 如果文件给出了 ent_rels 结果，则直接读取。否则需要临时查询
    if "ent_rels" in ap_info:
        for ep_str in ap_info["ent_rels"]:
            temp = ep_str.split(" ")
            ent = temp[0]
            tag = temp[1]
            rel = temp[2]
            if ent not in ent_rels:
                ent_rels[ent] = {"fwd": [], "rev": []}
            ent_rels[ent][tag].append(rel)
            if rel not in rel_rels:
                rel_rels[rel] = {"S-S": [], "S-O": [], "O-S": [], "O-O": []}
    else:
        for te in topic_ents:
            ent_rels[te] = {"fwd": [], "rev": []}
            neighbor_rels = FreebaseODBC.query_neighbor_rels([te])
            for rel in neighbor_rels:
                if rel.endswith("_Rev"):
                    rel = rel[:-4]
                    ent_rels[te]["rev"].append(rel)
                else:
                    ent_rels[te]["fwd"].append(rel)
                if rel not in rel_rels:
                    rel_rels[rel] = {"S-S": [], "S-O": [], "O-S": [], "O-O": []}
    # prepare rel_rel
    for pp_str in ap_info["candidates"]:
        splits = pp_str.split(" ")
        rel1 = "ns:" + splits[0]
        rel2 = "ns:" + splits[2]
        tag = splits[1]
        if rel1 not in rel_rels:
            rel_rels[rel1] = {"S-S": [], "S-O": [], "O-S": [], "O-O": []}
        rel_rels[rel1][tag].append(rel2)
    return ent_rels, rel_rels
