# Stardard Libraries
import json
import os
from typing import List, Dict, Tuple

# Self-defined Modules
from config import Config
from my_utils.freebase import FreebaseODBC
from my_utils.io_utils import append_jsonl, run_multiprocess
from my_utils.logger import Logger
from my_utils.pred_base import PredBase


class AdjacentInfoPrepare:
    logger = Logger.get_logger("AdjacentInfoPrepare", True)

    @classmethod
    def load_global_cache(cls):
        cls.logger.info(">>> prepare predicates' adjacent info...")
        processed_pred_set = set()
        preds_to_deal = []
        if os.path.exists(Config.cache_pred_conn):
            with open(Config.cache_pred_conn) as f:
                for line in f:
                    processed_pred_set.add(json.loads(line)["id"])
        cls.logger.debug(
            f"{len(processed_pred_set)} preds' adjacent info already in cache"
        )

        for pred in PredBase.pred_info_dict:
            if pred not in processed_pred_set:
                preds_to_deal.append(pred)
        cls.logger.debug(
            f"{len(preds_to_deal)} preds' adjacent info need to be queried"
        )

        # 多进程查询代码
        def mp_query_pred_conn(filepath, pid, queue):
            while True:
                pred = queue.get()
                if pred == None:
                    break
                info = FreebaseODBC.query_pred_conn(pred)
                res = {"id": pred, "pred_conn": info}
                cls.logger.debug(f"pid:{pid}, pred:{pred}")
                append_jsonl(res, filepath)

        run_multiprocess(
            mp_query_pred_conn,
            [Config.cache_pred_conn],
            preds_to_deal,
            Config.worker_num,
        )

        # 读取 cache
        cls.global_pred_pred_adjacent_info = dict()
        with open(Config.cache_pred_conn) as f:
            for line in f:
                temp = json.loads(line)
                cls.global_pred_pred_adjacent_info[temp["id"]] = temp["pred_conn"]

    @classmethod
    def get_ent_pred_adjacent_info(
        cls, ents: List[str], candi_preds: List[str]
    ) -> Dict[str, Dict[str, List[str]]]:
        ans = dict()
        for ent in ents:
            preds = set(candi_preds)
            ans[ent] = {"fwd": [], "rev": []}
            neighbor_preds = FreebaseODBC.query_neighbor_preds([ent])
            fwd_preds = list(
                set([pred for pred in neighbor_preds if not pred.endswith("_Rev")])
                & preds
            )
            rev_preds = list(
                set([pred[:-4] for pred in neighbor_preds if pred.endswith("_Rev")])
                & preds
            )
            ans[ent]["fwd"] = fwd_preds
            ans[ent]["rev"] = rev_preds
        return ans

    @classmethod
    def get_pred_pred_adjacent_info(
        cls, candi_preds: List[str]
    ) -> Dict[str, Dict[str, List[str]]]:
        # 缓存信息使用时才载入内存
        if cls.global_pred_pred_adjacent_info == None:
            cls.load_global_cache()
        ans = dict()
        candi_preds = set(candi_preds)
        for pred in candi_preds:
            ans[pred] = {"S-S": [], "S-O": [], "O-O": [], "O-S": []}
            if pred not in cls.global_pred_pred_adjacent_info:
                continue
            for tag in ["S-S", "S-O", "O-O", "O-S"]:
                for raw_pred in cls.global_pred_pred_adjacent_info[pred][tag]:
                    if raw_pred in candi_preds:
                        ans[pred][tag].append(raw_pred)
        return ans

    @classmethod
    def parse_adjacent_info_from_trips(
        cls, trips: List[List[str]], dedup=True
    ) -> Tuple[dict, dict]:
        # 在对相同 AP 重复数量敏感的场景，需要设置 dedup 为 False，避免相同角色重复的边被省略
        ent_conn_info = dict()
        for trip in trips:
            subj = trip[0]
            pred = trip[1]
            obj = trip[2]
            if dedup:
                if subj not in ent_conn_info:
                    ent_conn_info[subj] = {"subj": set(), "obj": set()}
                ent_conn_info[subj]["subj"].add(pred)
                if obj not in ent_conn_info:
                    ent_conn_info[obj] = {"subj": set(), "obj": set()}
                ent_conn_info[obj]["obj"].add(pred)
            else:
                if subj not in ent_conn_info:
                    ent_conn_info[subj] = {"subj": [], "obj": []}
                ent_conn_info[subj]["subj"].append(pred)
                if obj not in ent_conn_info:
                    ent_conn_info[obj] = {"subj": [], "obj": []}
                ent_conn_info[obj]["obj"].append(pred)
        # print(ent_conn_info)

        # collect pred-pred snippets
        pred_pred_snippets = dict()
        for ent in ent_conn_info:
            s_preds = list(ent_conn_info[ent]["subj"])
            o_preds = list(ent_conn_info[ent]["obj"])
            # Case1: S-S
            for i in range(len(s_preds)):
                for j in range(i + 1, len(s_preds)):
                    rel1 = s_preds[i]
                    rel2 = s_preds[j]
                    if rel1 not in pred_pred_snippets:
                        pred_pred_snippets[rel1] = {
                            "S-S": set(),
                            "S-O": set(),
                            "O-O": set(),
                            "O-S": set(),
                        }
                    if rel2 not in pred_pred_snippets:
                        pred_pred_snippets[rel2] = {
                            "S-S": set(),
                            "S-O": set(),
                            "O-O": set(),
                            "O-S": set(),
                        }
                    pred_pred_snippets[rel1]["S-S"].add(rel2)
                    pred_pred_snippets[rel2]["S-S"].add(rel1)
            # Case2: O-O
            for i in range(len(o_preds)):
                for j in range(i + 1, len(o_preds)):
                    rel1 = o_preds[i]
                    rel2 = o_preds[j]
                    if rel1 not in pred_pred_snippets:
                        pred_pred_snippets[rel1] = {
                            "S-S": set(),
                            "S-O": set(),
                            "O-O": set(),
                            "O-S": set(),
                        }
                    if rel2 not in pred_pred_snippets:
                        pred_pred_snippets[rel2] = {
                            "S-S": set(),
                            "S-O": set(),
                            "O-O": set(),
                            "O-S": set(),
                        }
                    pred_pred_snippets[rel1]["O-O"].add(rel2)
                    pred_pred_snippets[rel2]["O-O"].add(rel1)
            # Case3: S-O, O-S
            for i in range(len(s_preds)):
                for j in range(len(o_preds)):
                    rel1 = s_preds[i]
                    rel2 = o_preds[j]
                    if rel1 not in pred_pred_snippets:
                        pred_pred_snippets[rel1] = {
                            "S-S": set(),
                            "S-O": set(),
                            "O-O": set(),
                            "O-S": set(),
                        }
                    if rel2 not in pred_pred_snippets:
                        pred_pred_snippets[rel2] = {
                            "S-S": set(),
                            "S-O": set(),
                            "O-O": set(),
                            "O-S": set(),
                        }
                    pred_pred_snippets[rel1]["S-O"].add(rel2)
                    pred_pred_snippets[rel2]["O-S"].add(rel1)

        # collect ent-pred snippets
        ent_pred_snippets = dict()
        for ent in ent_conn_info:
            if ent.startswith("?"):
                continue
            else:
                ent_pred_snippets[ent] = {"fwd": [], "rev": []}
                for pred in ent_conn_info[ent]["subj"]:
                    ent_pred_snippets[ent]["fwd"].append(pred)
                for pred in ent_conn_info[ent]["obj"]:
                    ent_pred_snippets[ent]["rev"].append(pred)

        return ent_pred_snippets, pred_pred_snippets
