# Stardard Libraries
from copy import copy
from typing import Set, List, Dict

# Third party libraries

# Self-defined Modules
from config import Config
from my_utils.freebase import FreebaseODBC


""" 记录了组合过程中间状态信息的数据结构 """


class Combination:
    def __init__(self, ents: Set[str]) -> None:
        # node:{fwd_candi:set(), rev_candi:set(), bound_idx:int}
        self.node_info = dict()
        self.trips = []
        self.uncombined_ents = ents
        self.var_names = ["?x", "?y", "?z", "?k", "?i", "?m", "?n", "?j"]
        self.terminal_var = None

    # 需要负责所有相关变量的修改
    def add_trip(
        self, subj: str, rel: str, obj: str, adjacent_info: dict, rel_idx: int
    ):
        new_trip = [subj, rel, obj]
        # 处理 subject
        if subj == None:
            assert obj != None
            var_name = self.var_names.pop()
            new_trip[0] = var_name
            self.node_info[var_name] = {
                "fwd_candi": set(adjacent_info["S-S"]),
                "rev_candi": set(adjacent_info["S-O"]),
                "bound_idx": 0,
            }
            self.terminal_var = var_name
        elif subj.startswith("?"):
            assert (subj in self.node_info) and (
                rel in self.node_info[subj]["fwd_candi"]
            )  # and (rel_idx >= self.node_info[subj]["bound_idx"])
            self.node_info[subj]["fwd_candi"] = self.node_info[subj]["fwd_candi"] & set(
                adjacent_info["S-S"]
            )
            self.node_info[subj]["rev_candi"] = self.node_info[subj]["rev_candi"] & set(
                adjacent_info["S-O"]
            )
            self.node_info[subj]["bound_idx"] = rel_idx
            if subj == self.terminal_var:
                self.terminal_var = None
        else:
            assert (
                subj.startswith("ns:")
                and (subj not in self.node_info)
                and (subj in self.uncombined_ents)
            )
            self.uncombined_ents.remove(subj)
            self.node_info[subj] = {
                "fwd_candi": set(),
                "rev_candi": set(),
                "bound_idx": 0,
            }
        # 处理 object
        if obj == None:
            assert subj != None
            var_name = self.var_names.pop()
            new_trip[2] = var_name
            self.node_info[var_name] = {
                "fwd_candi": set(adjacent_info["O-S"]),
                "rev_candi": set(adjacent_info["O-O"]),
                "bound_idx": 0,
            }
            self.terminal_var = var_name
        elif obj.startswith("?"):
            assert (obj in self.node_info) and (
                rel in self.node_info[obj]["rev_candi"]
            )  # and (rel_idx >= self.node_info[obj]["bound_idx"])
            self.node_info[obj]["fwd_candi"] = self.node_info[obj]["fwd_candi"] & set(
                adjacent_info["O-S"]
            )
            self.node_info[obj]["rev_candi"] = self.node_info[obj]["rev_candi"] & set(
                adjacent_info["O-O"]
            )
            self.node_info[obj]["bound_idx"] = rel_idx
            if obj == self.terminal_var:
                self.terminal_var = None
        else:
            assert (
                obj.startswith("ns:")
                and (obj not in self.node_info)
                and (obj in self.uncombined_ents)
            )
            self.uncombined_ents.remove(obj)
            self.node_info[obj] = {
                "fwd_candi": set(),
                "rev_candi": set(),
                "bound_idx": 0,
            }
        # 更新 trips
        self.trips.append(new_trip)

    def get_attachable_vars(self) -> List[str]:
        if self.terminal_var != None:
            return [self.terminal_var]
        else:
            return [node for node in self.node_info if node.startswith("?")]

    @classmethod
    def __is_var_node(cls, node_name) -> bool:
        return node_name.startswith("?")

    def is_instantiable(self) -> bool:
        picked_one = None
        for node in self.node_info:
            if self.__is_var_node(node):
                picked_one = node
                break
        query = f"""SPARQL
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT {picked_one}
    WHERE {{
      {self.get_query_trips()}
    }}
    LIMIT 1
    """
        # print(query)
        return FreebaseODBC.has_query_results(query)

    def get_var_nodes(self) -> List[str]:
        ans = []
        for node in self.node_info:
            if self.__is_var_node(node):
                ans.append(node)
        return ans

    def get_instantiated_results(self, limit=1000) -> Dict[str, List[str]]:
        var_list = []
        for node in self.node_info:
            if self.__is_var_node(node):
                var_list.append(node)
        query = f"""SPARQL
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT {", ".join(var_list)}
    WHERE {{
      {self.get_query_trips()}
    }}
    LIMIT {limit}
    """
        return FreebaseODBC.get_query_results(query, var_list)

    def get_query_trips(self) -> str:
        temp = []
        for trip in self.trips:
            temp.append(" ".join(trip) + ".")
        return " ".join(temp)

    @classmethod
    def clone(cls, old_obj):
        new_obj = cls(set())
        new_obj.trips = copy(old_obj.trips)
        new_obj.var_names = copy(old_obj.var_names)
        new_obj.uncombined_ents = copy(old_obj.uncombined_ents)
        new_obj.terminal_var = old_obj.terminal_var
        for node in old_obj.node_info:
            old_info = old_obj.node_info[node]
            new_obj.node_info[node] = {
                "fwd_candi": copy(old_info["fwd_candi"]),
                "rev_candi": copy(old_info["rev_candi"]),
                "bound_idx": old_info["bound_idx"],
            }
        return new_obj


""" 控制 atomic patterns 的组合过程 """


class EPCombiner:
    max_combine_count = Config.max_combine_rels

    @classmethod
    def combine(
        cls,
        ent_rel_aps: dict,
        rel_rel_aps: dict,
        instantiable_check: bool = False,
    ) -> List[Combination]:
        assert len(ent_rel_aps) > 0
        ents = set(ent_rel_aps.keys())
        rel_idx = dict()
        for idx, rel in enumerate(
            cls.__collect_used_rels(ent_rel_aps, rel_rel_aps)
        ):
            rel_idx[rel] = idx
        init_combine = Combination(copy(ents))
        res = []
        # 选择邻接片段数量最少的实体作为组合起点
        start_ent = cls.__select_start_ent(ent_rel_aps, rel_rel_aps)
        trips = []
        for rel in ent_rel_aps[start_ent]["fwd"]:
            trips.append((start_ent, rel, None))
        for rel in ent_rel_aps[start_ent]["rev"]:
            trips.append((None, rel, start_ent))
        for trip in trips:
            temp = Combination.clone(init_combine)
            adjacent_info = {"S-S": set(), "S-O": set(), "O-S": set(), "O-O": set()}
            if trip[1] in rel_rel_aps:
                adjacent_info = rel_rel_aps[trip[1]]
            temp.add_trip(trip[0], trip[1], trip[2], adjacent_info, rel_idx[trip[1]])
            if cls.__meet_candi_cond(temp):
                res.append(temp)
            if cls.__meet_expand_cond(temp):
                cls.__combine(
                    ent_rel_aps,
                    rel_rel_aps,
                    rel_idx,
                    temp,
                    res,
                    instantiable_check,
                )
        return res

    @classmethod
    def __collect_used_rels(cls, ent_rels: dict, rel_rels: dict) -> Set[str]:
        ans = set()
        for ent in ent_rels:
            ans |= set(ent_rels[ent]["fwd"])
            ans |= set(ent_rels[ent]["rev"])
        ans |= rel_rels.keys()
        return ans

    @classmethod
    def __meet_candi_cond(cls, combi: Combination) -> bool:
        return len(combi.uncombined_ents) == 0

    @classmethod
    def __meet_expand_cond(cls, combi: Combination) -> bool:
        return len(combi.trips) < cls.max_combine_count

    @classmethod
    def __select_start_ent(
        cls, ent_rel_aps: dict, rel_rel_aps: dict
    ) -> str:
        ents_score = []
        for ent in ent_rel_aps:
            count = 0
            for rel in ent_rel_aps[ent]["fwd"]:
                if rel in rel_rel_aps:
                    count += len(rel_rel_aps[rel]["O-S"])
                    count += len(rel_rel_aps[rel]["O-O"])
            for rel in ent_rel_aps[ent]["rev"]:
                if rel in rel_rel_aps:
                    count += len(rel_rel_aps[rel]["S-S"])
                    count += len(rel_rel_aps[rel]["S-O"])
            ents_score.append((ent, count))
        ents_score.sort(key=lambda x: x[1])
        return ents_score[0][0]

    @classmethod
    def __combine(
        cls,
        ent_rel_aps: dict,
        rel_rel_aps: dict,
        rel_idx: dict,
        current_combi: Combination,
        res: list,
        instantiable_check: bool = False,
    ):
        # Step1: 提取出当前 combination 的变量节点信息和尚未处理的约束节点
        node_info = current_combi.node_info
        uncombined_ents = current_combi.uncombined_ents
        # Step2: 遍历变量节点信息，枚举可以拓展的候选三元组（同时考虑变量和约束）
        candi_trip = []
        for node in current_combi.get_attachable_vars():
            fwd_candi = node_info[node]["fwd_candi"]
            rev_candi = node_info[node]["rev_candi"]
            bound_idx = node_info[node]["bound_idx"]
            # 遍历检查正向候选谓词
            for rel in fwd_candi:
                # 为了规避重复组合的情况，要求在每个变量节点添加的 rel idx 具有不递减的倾向 (初始为0)
                # if rel_idx[rel] < bound_idx:
                #   continue
                candi_trip.append((node, rel, None))
                # 考虑同时添加一个节点约束的情况
                topic_ents = cls.__choose_candi_topic_ents(
                    uncombined_ents, ent_rel_aps, rel, "rev"
                )
                for ent in topic_ents:
                    candi_trip.append((node, rel, ent))
            # 遍历检查逆向候选谓词 (同理)
            for rel in rev_candi:
                # if rel_idx[rel] < bound_idx:
                #   continue
                candi_trip.append((None, rel, node))
                topic_ents = cls.__choose_candi_topic_ents(
                    uncombined_ents, ent_rel_aps, rel, "fwd"
                )
                for ent in topic_ents:
                    candi_trip.append((ent, rel, node))
        # Step3: 根据候选三元组对当前组合进行基础判定和拓展
        for trip in candi_trip:
            temp = Combination.clone(current_combi)
            temp.add_trip(
                trip[0],
                trip[1],
                trip[2],
                rel_rel_aps[trip[1]],
                rel_idx[trip[1]],
            )
            candi_flag = cls.__meet_candi_cond(temp)
            expand_flag = cls.__meet_expand_cond(temp)
            # 在组合的过程中进行实例化剪枝
            if (
                instantiable_check
                and (candi_flag or expand_flag)
                and not temp.is_instantiable()
            ):
                continue
            if candi_flag:
                res.append(temp)
            if expand_flag:
                cls.__combine(
                    ent_rel_aps,
                    rel_rel_aps,
                    rel_idx,
                    temp,
                    res,
                    instantiable_check,
                )
        # print(len(res))

    @classmethod
    def __choose_candi_topic_ents(
        cls, candi_ents: Set[str], ent_rel_aps: dict, target_rel: str, dir: str
    ):
        ans = []
        for ent in candi_ents:
            for rel in ent_rel_aps[ent][dir]:
                if rel == target_rel:
                    ans.append(ent)
                    break
        return ans
