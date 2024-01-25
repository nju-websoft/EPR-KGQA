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


def update_base(predFile: str, typeFile: str):
    freebase = FreebaseODBC()
    pred_info_dict = freebase.query_pred_info()
    # 有些谓词的逆向未记录完整（只记录了单向，这里补上）
    count = 0
    for prop in pred_info_dict:
        item = pred_info_dict[prop]
        rev = item["reverse"]
        if rev == None:
            continue
        if rev in pred_info_dict and pred_info_dict[rev]["reverse"] == None:
            pred_info_dict[rev]["reverse"] = prop
            count += 1
    print(f"Add {count} missing reverse properties!")
    with open(predFile, "w") as f:
        json.dump(pred_info_dict, f, indent=2, ensure_ascii=False)
    # 搜集类型的基本信息 (name, #inst, #inst_with_name)
    type_set = set()
    for prop in pred_info_dict:
        d = pred_info_dict[prop]["domain"]
        r = pred_info_dict[prop]["range"]
        if d:
            type_set.add(d)
        if r:
            type_set.add(r)
    type_info_dict = freebase.query_type_info(type_set)
    with open(typeFile, "w") as f:
        json.dump(type_info_dict, f, indent=2, ensure_ascii=False)


class PredBase:
    logger = Logger.get_logger("PredBase", True)
    if (not os.path.exists(Config.cache_pred_info)) or (
        not os.path.exists(Config.cache_type_info)
    ):
        logger.info(">>> no cache exists, query predicate/type info...")
        update_base(Config.cache_pred_info, Config.cache_type_info)
    else:
        logger.info(">>> read predicate/type info from cache file...")
    pred_info_dict = json.load(open(Config.cache_pred_info))
    type_info_dict = json.load(open(Config.cache_type_info))
    logger.debug(f"{len(pred_info_dict)} predicates and {len(type_info_dict)} types")
    stop_words = set(stopwords.words("english"))

    @classmethod
    def get_reverse(cls, pred: str) -> str:
        if pred not in cls.pred_info_dict:
            return None
        else:
            return cls.pred_info_dict[pred]["reverse"]

    @classmethod
    def get_domain(cls, pred: str):
        if pred not in cls.pred_info_dict:
            return None
        else:
            return cls.pred_info_dict[pred]["domain"]

    @classmethod
    def get_range(cls, pred: str):
        if pred not in cls.pred_info_dict:
            return None
        else:
            return cls.pred_info_dict[pred]["range"]

    @classmethod
    def get_label(cls, pred: str):
        if pred not in cls.pred_info_dict:
            return None
        else:
            return cls.pred_info_dict[pred]["name"]

    @classmethod
    def get_same_form(cls, pred: str, reverse=False):
        """把谓词统一成一种确定的形态, 人为规定逆向与否"""
        if pred.endswith("_Rev"):
            if reverse:
                return pred
            else:
                temp = pred.replace("_Rev", "")
                return cls.get_reverse(temp)
        else:
            if not reverse:
                return pred
            else:
                rev = cls.get_reverse(pred)
                if rev == None:
                    return None
                else:
                    return rev + "_Rev"

    @classmethod
    def is_reverse_pair(cls, pred1: str, pred2: str) -> bool:
        # relation_Rev 与 relation 的直接判定
        if (
            pred1.endswith("_Rev")
            and (not pred2.endswith("_Rev"))
            and pred1[:-4] == pred2
        ):
            return True
        elif (
            pred2.endswith("_Rev")
            and (not pred1.endswith("_Rev"))
            and pred2[:-4] == pred1
        ):
            return True
        # 取逆关系进行判定
        else:
            pred1 = cls.get_same_form(pred1)
            pred2 = cls.get_same_form(pred2)
            if pred1 == None or pred2 == None:
                return False
            if cls.get_reverse(pred1) == pred2:
                return True
        return False

    @classmethod
    def get_pred_set(cls) -> set:
        return set(cls.pred_info_dict.keys())

    @classmethod
    def get_relevant_preds(cls, pred: str) -> set:
        """relevant 定义: 具有相同的 domain/range/label"""
        if not pred.startswith("ns:"):
            pred = "ns:" + pred
        feature_set = set()
        if cls.get_domain(pred):
            feature_set.add(cls.get_domain(pred))
        if cls.get_range(pred):
            feature_set.add(cls.get_range(pred))
        if cls.get_label(pred):
            feature_set.add(cls.get_label(pred))
        ans_set = set()
        for p in cls.pred_info_dict:
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
    def reach_cvt(cls, pred: str) -> bool:
        target_type = None
        if pred.endswith("_Rev"):
            target_type = cls.get_range(cls.get_same_form(pred))
        else:
            target_type = cls.get_range(pred)
        return cls.is_cvt_type(target_type)

    @classmethod
    def from_cvt(cls, pred: str) -> bool:
        target_type = None
        if pred.endswith("_Rev"):
            target_type = cls.get_domain(cls.get_same_form(pred))
        else:
            target_type = cls.get_domain(pred)
        return cls.is_cvt_type(target_type)

    @classmethod
    def is_nary_pred(cls, pred: str) -> bool:
        if pred.endswith("_Rev"):
            pred = cls.get_same_form(pred)
        domain = cls.get_domain(pred)
        range = cls.get_range(pred)
        return cls.is_cvt_type(domain) or cls.is_cvt_type(range)

    @classmethod
    def get_pred_keywords(cls, pred: str) -> set:
        word_set = set()
        for s in pred.split(".")[-2:]:
            for word in s.split("_"):
                if word not in cls.stop_words:
                    word_set.add(word)
        return word_set

    @classmethod
    def get_rich_pred_str(cls, pred: str) -> set:
        """拓展 predicate 信息： predicate|label|domain|range|revLabel"""
        if not pred.startswith("ns:"):
            pred = "ns:" + pred
        rich_relation = ""
        rich_relation += pred.replace("ns:", "")
        # rich_relation += " | "
        # rich_relation += str(cls.get_label(pred))
        # rich_relation += " | "
        # rich_relation += str(cls.get_domain(pred)).replace("ns:","")
        # rich_relation += " | "
        # rich_relation += str(cls.get_range(pred)).replace("ns:","")
        # rich_relation += " | "
        # rich_relation += str(cls.get_label(cls.get_reverse(pred)))
        return rich_relation
