# Stardard Libraries
import re
from typing import *

# Third party libraries
from tqdm import tqdm

# Self-defined Modules
from my_utils.rel_base import relBase


def filter_topk_rels(ranked_rel_list: List[str], topk: int, deduplicate: bool) -> List[str]:
    # 预处理，确保 relation 都有前缀
    for idx, rel in enumerate(ranked_rel_list):
        if not rel.startswith("ns:"):
            ranked_rel_list[idx] = "ns:" + rel
    temp_list = []
    temp_set = set()
    if deduplicate:
        for rel in ranked_rel_list:
            if relBase.get_reverse(rel) in temp_set:
                continue
            temp_list.append(rel)
            temp_set.add(rel)
    else:
        temp_list = ranked_rel_list
    return temp_list[:topk]


def get_rr_aps_str_set(rr_aps_info: dict) -> Set[str]:
    ans = set()
    for item1 in rr_aps_info:
        for tag in rr_aps_info[item1]:
            for item2 in rr_aps_info[item1][tag]:
                ans.add(" ".join([item1, item2, tag]))
    return ans


def expand_ent_rel_aps_str_set(ent: str, rel: str, tag: str) -> Set[str]:
    trans_dict = {"rev": "fwd", "fwd": "rev"}
    ans = set()
    ans.add(" ".join([ent, rel, tag]))
    rev = relBase.get_reverse(rel)
    if rev != None:
        new_tag = trans_dict[tag]
        ans.add(" ".join([ent, rev, new_tag]))
    return ans


def expand_rel_rel_aps_str_set(rel1: str, rel2: str, tag: str) -> Set[str]:
    trans_dict = {"S": "O", "O": "S"}
    ans = set()
    ans.add(" ".join([rel1, rel2, tag]))
    rev1 = relBase.get_reverse(rel1)
    rev2 = relBase.get_reverse(rel2)
    if rev2 != None:
        ans.add(" ".join([rel1, rev2, tag[0] + "-" + trans_dict[tag[2]]]))
    if rev1 != None:
        ans.add(" ".join([rev1, rel2, trans_dict[tag[0]] + "-" + tag[2]]))
    if rev1 != None and rev2 != None:
        ans.add(" ".join([rev1, rev2, trans_dict[tag[0]] + "-" + trans_dict[tag[2]]]))
    return ans


def cover_gold_rr_aps(rel_rels, gold_rel_rels, check_rel=True) -> bool:
    res = True
    target_aps_str_set = set()
    target_aps_str_set |= get_rr_aps_str_set(rel_rels)
    if check_rel and res:
        for rel1 in gold_rel_rels:
            for tag in gold_rel_rels[rel1]:
                for rel2 in gold_rel_rels[rel1][tag]:
                    res = res and len(target_aps_str_set & expand_rel_rel_aps_str_set(rel1, rel2, tag)) != 0
    return res


def parse_triplets_from_serialized_ep(ep: str) -> List[List[str]]:
    tripstrs = re.findall(r"[\?n][^ ()?{}]+ ns:[^ ]+ [\?n][^ ()?{}]+", ep)
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


def accumulate_rel_rel_statistics(
    rel_rels: Dict[str, Dict[str, List[str]]], statistics: Dict[str, float]
):
    if len(rel_rels) == 0:
        return
    tags = ["S-S", "S-O", "O-O", "O-S"]
    temp = {"num": 0, "S-S": 0, "S-O": 0, "O-O": 0, "O-S": 0}
    for rel in rel_rels:
        temp["num"] += 1
        for tag in tags:
            temp[tag] += len(rel_rels[rel][tag])
    statistics["num"] += temp["num"]
    for tag in tags:
        statistics[tag] += temp[tag] / temp["num"]


def normalize_triplet(triplet: List[str]) -> List[str]:
    if triplet[1].endswith("_Rev"):
        return [triplet[2], triplet[1][:-4], triplet[0]]
    else:
        return triplet


def parse_info_from_ranked_rr_aps(
    ranked_rr_aps: List[str], ranked_rels: List[str]
) -> Dict[str, Dict[str, List[str]]]:
    tag_map = {"S-S": "S-S", "S-O": "O-S", "O-S": "S-O", "O-O": "O-O"}
    ans = dict()
    for rel in ranked_rels:
        ans[rel] = {"S-S": set(), "S-O": set(), "O-S": set(), "O-O": set()}
    for snp in ranked_rr_aps:
        temp = snp.split(" ")
        p1 = "ns:" + temp[0]
        tag = temp[1]
        p2 = "ns:" + temp[2]
        ans[p1][tag].add(p2)
        ans[p2][tag_map[tag]].add(p1)
    return ans


def parse_aps_dict_from_strs(
    topic_ents: List[str], ent_rel_strs: List[str], rel_rel_strs: List[str]
) -> Tuple[dict, dict]:
    pp_tag_map = {"S-S": "S-S", "S-O": "O-S", "O-S": "S-O", "O-O": "O-O"}
    ent_rels = dict()
    rel_rels = dict()
    for ent in topic_ents:
        ent_rels[ent] = {"fwd": set(), "rev": set()}

    for ep in ent_rel_strs:
        temp = ep.split(" ")
        ent = "ns:" + temp[0]
        tag = temp[1]
        rel = "ns:" + temp[2]
        if ent not in ent_rels:
            ent_rels[ent] = {"fwd": set(), "rev": set()}
        ent_rels[ent][tag].add(rel)
        # 补充 pp 中的信息
        if rel not in rel_rels:
            rel_rels[rel] = {"S-S": set(), "S-O": set(), "O-S": set(), "O-O": set()}

    for pp in rel_rel_strs:
        temp = pp.split(" ")
        p1 = "ns:" + temp[0]
        tag = temp[1]
        p2 = "ns:" + temp[2]
        if p1 not in rel_rels:
            rel_rels[p1] = {"S-S": set(), "S-O": set(), "O-S": set(), "O-O": set()}
        if p2 not in rel_rels:
            rel_rels[p2] = {"S-S": set(), "S-O": set(), "O-S": set(), "O-O": set()}
        rel_rels[p1][tag].add(p2)
        rel_rels[p2][pp_tag_map[tag]].add(p1)

    return ent_rels, rel_rels


def filter_topk_aps(ranked_aps: List[dict], topk: int) -> List[dict]:
    """merge equivalent aps && filter topk && expand ap by tag
    e.g. [(p1,p2,S-S), (p1_rev,p3,S-O), (p2_rev,p3_rev,O-O)] -> [(p1,p2,S-S), (p1,p3,O-O), (P2,P3,S-S)]
    [param] ranked_aps: [{id:str, split_id:str, question:str, candidates:[str,...]}...]
    """
    trans_dict = {"S": "O", "O": "S"}
    for item in tqdm(ranked_aps):
        used_rels = set()
        topk_aps = set()
        expanded_aps = set()
        # merge reversed-equivalent aps && filter topk
        for ap in item["rr_aps"]:
            if len(topk_aps) >= topk:
                break
            temp = ap.split(" ")
            rel1 = temp[0]
            tag = temp[1]
            rel2 = temp[2]
            rev1 = relBase.get_reverse("ns:" + rel1)
            rev2 = relBase.get_reverse("ns:" + rel2)
            if rev1 != None and rev1[3::] in used_rels:
                rel1 = rev1[3::]
                tag = trans_dict[tag[0]] + tag[1] + tag[2]
            used_rels.add(rel1)
            if rev2 != None and rev2[3::] in used_rels:
                rel2 = rev2[3::]
                tag = tag[0] + tag[1] + trans_dict[tag[2]]
            used_rels.add(rel2)
            topk_aps.add(" ".join([rel1, tag, rel2]))
            # expand ap by tag
            new_tag = tag[2] + tag[1] + tag[0]
            expanded_aps.add(" ".join([rel2, new_tag, rel1]))
        item["rr_aps"] = list(topk_aps | expanded_aps)
    return ranked_aps
