# Stardard Libraries
import json
import os
from typing import List, Dict, Tuple

# Self-defined Modules
from config import Config
from my_utils.freebase import FreebaseODBC
from my_utils.io_utils import append_jsonl, run_multiprocess
from my_utils.logger import Logger
from my_utils.rel_base import relBase


class AdjacentInfoPrepare:
    if not os.path.exists(Config.cache_dir):
        os.makedirs(Config.cache_dir)
    logger = Logger.get_logger("AdjacentInfoPrepare", True)

    @classmethod
    def load_global_cache(cls):
        cls.logger.info(">>> prepare relations' adjacent info...")
        processed_rel_set = set()
        rels_to_deal = []
        if os.path.exists(Config.cache_rel_conn):
            with open(Config.cache_rel_conn) as f:
                for line in f:
                    processed_rel_set.add(json.loads(line)["id"])
        cls.logger.debug(
            f"{len(processed_rel_set)} rels' adjacent info already in cache"
        )

        for rel in relBase.rel_info_dict:
            if rel not in processed_rel_set:
                rels_to_deal.append(rel)
        cls.logger.debug(
            f"{len(rels_to_deal)} rels' adjacent info need to be queried"
        )

        # 多进程查询代码
        def mp_query_rel_conn(filepath, pid, queue):
            while True:
                rel = queue.get()
                if rel == None:
                    break
                info = FreebaseODBC.query_rel_conn(rel)
                res = {"id": rel, "rel_conn": info}
                cls.logger.debug(f"pid:{pid}, rel:{rel}")
                append_jsonl(res, filepath)

        run_multiprocess(
            mp_query_rel_conn,
            [Config.cache_rel_conn],
            rels_to_deal,
            Config.worker_num,
        )

        # 读取 cache
        cls.global_rel_rel_adjacent_info = dict()
        with open(Config.cache_rel_conn) as f:
            for line in f:
                temp = json.loads(line)
                cls.global_rel_rel_adjacent_info[temp["id"]] = temp["rel_conn"]


    @classmethod
    def parse_adjacent_info_from_trips(
        cls, trips: List[List[str]], dedup=True
    ) -> Tuple[dict, dict]:
        # 在对相同 AP 重复数量敏感的场景，需要设置 dedup 为 False，避免相同角色重复的边被省略
        ent_conn_info = dict()
        for trip in trips:
            subj = trip[0]
            rel = trip[1]
            obj = trip[2]
            if dedup:
                if subj not in ent_conn_info:
                    ent_conn_info[subj] = {"subj": set(), "obj": set()}
                ent_conn_info[subj]["subj"].add(rel)
                if obj not in ent_conn_info:
                    ent_conn_info[obj] = {"subj": set(), "obj": set()}
                ent_conn_info[obj]["obj"].add(rel)
            else:
                if subj not in ent_conn_info:
                    ent_conn_info[subj] = {"subj": [], "obj": []}
                ent_conn_info[subj]["subj"].append(rel)
                if obj not in ent_conn_info:
                    ent_conn_info[obj] = {"subj": [], "obj": []}
                ent_conn_info[obj]["obj"].append(rel)
        # print(ent_conn_info)

        # collect rel-rel aps
        rel_rel_aps = dict()
        for ent in ent_conn_info:
            s_rels = list(ent_conn_info[ent]["subj"])
            o_rels = list(ent_conn_info[ent]["obj"])
            # Case1: S-S
            for i in range(len(s_rels)):
                for j in range(i + 1, len(s_rels)):
                    rel1 = s_rels[i]
                    rel2 = s_rels[j]
                    if rel1 not in rel_rel_aps:
                        rel_rel_aps[rel1] = {
                            "S-S": set(),
                            "S-O": set(),
                            "O-O": set(),
                            "O-S": set(),
                        }
                    if rel2 not in rel_rel_aps:
                        rel_rel_aps[rel2] = {
                            "S-S": set(),
                            "S-O": set(),
                            "O-O": set(),
                            "O-S": set(),
                        }
                    rel_rel_aps[rel1]["S-S"].add(rel2)
                    rel_rel_aps[rel2]["S-S"].add(rel1)
            # Case2: O-O
            for i in range(len(o_rels)):
                for j in range(i + 1, len(o_rels)):
                    rel1 = o_rels[i]
                    rel2 = o_rels[j]
                    if rel1 not in rel_rel_aps:
                        rel_rel_aps[rel1] = {
                            "S-S": set(),
                            "S-O": set(),
                            "O-O": set(),
                            "O-S": set(),
                        }
                    if rel2 not in rel_rel_aps:
                        rel_rel_aps[rel2] = {
                            "S-S": set(),
                            "S-O": set(),
                            "O-O": set(),
                            "O-S": set(),
                        }
                    rel_rel_aps[rel1]["O-O"].add(rel2)
                    rel_rel_aps[rel2]["O-O"].add(rel1)
            # Case3: S-O, O-S
            for i in range(len(s_rels)):
                for j in range(len(o_rels)):
                    rel1 = s_rels[i]
                    rel2 = o_rels[j]
                    if rel1 not in rel_rel_aps:
                        rel_rel_aps[rel1] = {
                            "S-S": set(),
                            "S-O": set(),
                            "O-O": set(),
                            "O-S": set(),
                        }
                    if rel2 not in rel_rel_aps:
                        rel_rel_aps[rel2] = {
                            "S-S": set(),
                            "S-O": set(),
                            "O-O": set(),
                            "O-S": set(),
                        }
                    rel_rel_aps[rel1]["S-O"].add(rel2)
                    rel_rel_aps[rel2]["O-S"].add(rel1)

        # collect ent-rel aps
        ent_rel_aps = dict()
        for ent in ent_conn_info:
            if ent.startswith("?"):
                continue
            else:
                ent_rel_aps[ent] = {"fwd": [], "rev": []}
                for rel in ent_conn_info[ent]["subj"]:
                    ent_rel_aps[ent]["fwd"].append(rel)
                for rel in ent_conn_info[ent]["obj"]:
                    ent_rel_aps[ent]["rev"].append(rel)

        return ent_rel_aps, rel_rel_aps
