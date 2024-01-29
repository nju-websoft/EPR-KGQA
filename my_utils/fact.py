from enum import Enum
from typing import List


class Fact:
    class TFact(Enum):
        BINARY = 1
        NARY = 2

    # label_service

    def __init__(self, triplets: List[List[str]]) -> None:
        self.triplets = triplets
        self.serialize_str = self.__serialize()

    def __str__(self) -> str:
        return self.serialize_str

    def __repr__(self) -> str:
        return self.serialize_str

    def __hash__(self) -> int:
        return hash(self.serialize_str)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fact):
            return False
        return self.serialize_str == other.serialize_str

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __serialize(self) -> str:
        ans = []
        for trip in self.triplets:
            ans.append(" ".join(trip) + ".")
        ans = list(set(ans))
        ans.sort()
        return "Fact:{" + "\t".join(ans) + "}"

    @classmethod
    def init_by_factstr(cls, serialize_str: str):
        tripstrs = serialize_str[6:-2].split(".\t")
        trips = []
        for tripstr in tripstrs:
            trips.append(tripstr.split(" "))
        return cls(trips)

    def get_trip_count(self) -> int:
        return len(self.triplets)

    def get_fact_type(self) -> TFact:
        assert len(self.triplets) > 0
        if len(self.triplets) == 1:
            return Fact.TFact.BINARY
        else:
            return Fact.TFact.NARY

    def is_binary_fact(self) -> bool:
        return self.get_fact_type() == Fact.TFact.BINARY

    def is_nary_fact(self) -> bool:
        return self.get_fact_type() == Fact.TFact.NARY

    def get_cvt_ent(self):
        # works for nary fact only
        assert self.is_nary_fact()
        trip1 = self.triplets[0]
        trip2 = self.triplets[1]
        if trip1[0] == trip2[0] or trip1[0] == trip2[2]:
            return trip1[0]
        else:
            return trip1[2]

    def get_nl_str(self, use_paraphrase: bool = False):
        pass

    def get_abs_info(self):
        # TODO: 对于时间要素的处理
        if self.is_binary_fact():
            pred = self.triplets[0][1]
            subj = self.triplets[0][0]
            obj = self.triplets[0][2]
            temp1 = {"pred": pred, "role": "subj", "value": [subj]}
            temp2 = {"pred": pred, "role": "obj", "value": [obj]}
            return [temp1, temp2]
        elif self.is_nary_fact():
            cvt = self.get_cvt_ent()
            pred_info = dict()
            for trip in self.triplets:
                subj = trip[0]
                pred = trip[1]
                obj = trip[2]
                if pred not in pred_info:
                    pred_info[pred] = {"subj": [], "obj": []}
                if subj != cvt:
                    pred_info[pred]["subj"].append(subj)
                if obj != cvt:
                    pred_info[pred]["obj"].append(obj)
            ans = []
            for pred in pred_info:
                if len(pred_info[pred]["subj"]) > 0:
                    ans.append(
                        {"pred": pred, "role": "subj", "value": pred_info[pred]["subj"]}
                    )
                if len(pred_info[pred]["obj"]) > 0:
                    ans.append(
                        {"pred": pred, "role": "obj", "value": pred_info[pred]["obj"]}
                    )
            return ans
        return None
