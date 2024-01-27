# Stardard Libraries
import json
import os

# Third party libraries
from nltk.corpus import stopwords

# Self-defined Modules
from config import Config
from my_utils.freebase import FreebaseODBC
from my_utils.logger import Logger


UPDATE_CONN = True


def update_base(relFile: str, typeFile: str):
    freebase = FreebaseODBC()
    rel_info_dict = freebase.query_rel_info()
    # 有些谓词的逆向未记录完整（只记录了单向，这里补上）
    count = 0
    for prop in rel_info_dict:
        item = rel_info_dict[prop]
        rev = item["reverse"]
        if rev == None:
            continue
        if rev in rel_info_dict and rel_info_dict[rev]["reverse"] == None:
            rel_info_dict[rev]["reverse"] = prop
            count += 1
    print(f"Add {count} missing reverse properties!")
    with open(relFile, "w") as f:
        json.dump(rel_info_dict, f, indent=2, ensure_ascii=False)
    # 搜集类型的基本信息 (name, #inst, #inst_with_name)
    type_set = set()
    for prop in rel_info_dict:
        d = rel_info_dict[prop]["domain"]
        r = rel_info_dict[prop]["range"]
        if d:
            type_set.add(d)
        if r:
            type_set.add(r)
    type_info_dict = freebase.query_type_info(type_set)
    with open(typeFile, "w") as f:
        json.dump(type_info_dict, f, indent=2, ensure_ascii=False)


class relBase:
    if not os.path.exists(Config.cache_dir):
        os.makedirs(Config.cache_dir)
    logger = Logger.get_logger("relBase", True)
    if (not os.path.exists(Config.cache_rel_info)) or (
        not os.path.exists(Config.cache_type_info)
    ):
        logger.info(">>> no cache exists, query relation/type info...")
        update_base(Config.cache_rel_info, Config.cache_type_info)
    else:
        logger.info(">>> read relation/type info from cache file...")
    rel_info_dict = json.load(open(Config.cache_rel_info))
    type_info_dict = json.load(open(Config.cache_type_info))
    logger.debug(f"{len(rel_info_dict)} relations and {len(type_info_dict)} types")
    stop_words = set(stopwords.words("english"))

    @classmethod
    def get_reverse(cls, rel: str) -> str:
        if rel not in cls.rel_info_dict:
            return None
        else:
            return cls.rel_info_dict[rel]["reverse"]

    @classmethod
    def get_domain(cls, rel: str):
        if rel not in cls.rel_info_dict:
            return None
        else:
            return cls.rel_info_dict[rel]["domain"]

    @classmethod
    def get_range(cls, rel: str):
        if rel not in cls.rel_info_dict:
            return None
        else:
            return cls.rel_info_dict[rel]["range"]

    @classmethod
    def get_label(cls, rel: str):
        if rel not in cls.rel_info_dict:
            return None
        else:
            return cls.rel_info_dict[rel]["name"]

    @classmethod
    def get_same_form(cls, rel: str, reverse=False):
        """把谓词统一成一种确定的形态, 人为规定逆向与否"""
        if rel.endswith("_Rev"):
            if reverse:
                return rel
            else:
                temp = rel.replace("_Rev", "")
                return cls.get_reverse(temp)
        else:
            if not reverse:
                return rel
            else:
                rev = cls.get_reverse(rel)
                if rev == None:
                    return None
                else:
                    return rev + "_Rev"

    @classmethod
    def is_reverse_pair(cls, rel1: str, rel2: str) -> bool:
        # relation_Rev 与 relation 的直接判定
        if (
            rel1.endswith("_Rev")
            and (not rel2.endswith("_Rev"))
            and rel1[:-4] == rel2
        ):
            return True
        elif (
            rel2.endswith("_Rev")
            and (not rel1.endswith("_Rev"))
            and rel2[:-4] == rel1
        ):
            return True
        # 取逆关系进行判定
        else:
            rel1 = cls.get_same_form(rel1)
            rel2 = cls.get_same_form(rel2)
            if rel1 == None or rel2 == None:
                return False
            if cls.get_reverse(rel1) == rel2:
                return True
        return False

    @classmethod
    def get_rel_set(cls) -> set:
        return set(cls.rel_info_dict.keys())

    @classmethod
    def get_relevant_rels(cls, rel: str) -> set:
        """relevant 定义: 具有相同的 domain/range/label"""
        if not rel.startswith("ns:"):
            rel = "ns:" + rel
        feature_set = set()
        if cls.get_domain(rel):
            feature_set.add(cls.get_domain(rel))
        if cls.get_range(rel):
            feature_set.add(cls.get_range(rel))
        if cls.get_label(rel):
            feature_set.add(cls.get_label(rel))
        ans_set = set()
        for p in cls.rel_info_dict:
            fset = set()
            if cls.get_domain(p):
                fset.add(cls.get_domain(p))
            if cls.get_range(p):
                fset.add(cls.get_range(p))
            if cls.get_label(p):
                fset.add(cls.get_label(p))
            if len(fset & feature_set) != 0:
                ans_set.add(p)
        return set([p.replace("ns:", "") for p in ans_set])

    @classmethod
    def is_cvt_type(cls, tp: str) -> bool:
        if tp == None:
            return False
        if cls.type_info_dict[tp]["instance_count"] == 0:
            return False
        return (
            cls.type_info_dict[tp]["has_name_instance_count"]
            * 1.0
            / cls.type_info_dict[tp]["instance_count"]
        ) < 0.01

    @classmethod
    def reach_cvt(cls, rel: str) -> bool:
        target_type = None
        if rel.endswith("_Rev"):
            target_type = cls.get_range(cls.get_same_form(rel))
        else:
            target_type = cls.get_range(rel)
        return cls.is_cvt_type(target_type)

    @classmethod
    def from_cvt(cls, rel: str) -> bool:
        target_type = None
        if rel.endswith("_Rev"):
            target_type = cls.get_domain(cls.get_same_form(rel))
        else:
            target_type = cls.get_domain(rel)
        return cls.is_cvt_type(target_type)

    @classmethod
    def is_nary_rel(cls, rel: str) -> bool:
        if rel.endswith("_Rev"):
            rel = cls.get_same_form(rel)
        domain = cls.get_domain(rel)
        range = cls.get_range(rel)
        return cls.is_cvt_type(domain) or cls.is_cvt_type(range)

    @classmethod
    def get_rel_keywords(cls, rel: str) -> set:
        word_set = set()
        for s in rel.split(".")[-2:]:
            for word in s.split("_"):
                if word not in cls.stop_words:
                    word_set.add(word)
        return word_set
