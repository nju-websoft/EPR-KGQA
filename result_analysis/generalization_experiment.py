# Stardard Libraries
import json
import os
import re

# Third party libraries
from tqdm import tqdm

# Self-defined Modules
from config import Config
from my_utils.data_item import load_ds_items


def generate_cwq_relations_in_train_set():
    if os.path.exists(Config.ds_train_relations):
        print(f"{Config.ds_train_relations} already exists!")
        return
    with open(Config.ds_train, "r") as f:
        train_data = json.load(f)
    train_rels_set = set()
    for idx, data_item in enumerate(tqdm(train_data)):
        sparql = data_item["sparql"]
        pattern = r"ns:\S+"
        matches = re.findall(pattern, sparql)
        for match in matches:
            if match.startswith("ns:m.") or match.startswith("ns:g."):
                continue
            match = match[3:]
            train_rels_set.add(match)
    with open(Config.ds_train_relations, "w") as f:
        json.dump(list(train_rels_set), f)
    print(1)


def split_cwq_test_by_iid_and_zero_shot():
    if os.path.exists(Config.ds_test_iid_idxs) and os.path.exists(
        Config.ds_test_zero_shot_idxs
    ):
        print(f"{Config.ds_test_iid_idxs} and {Config.ds_test_zero_shot_idxs} exist!")
        return
    with open(Config.ds_test, "r") as f:
        test_data = json.load(f)
    with open(Config.ds_train_relations, "r") as f:
        train_rels_set = set(json.load(f))
    test_iid_idxs = []
    test_zero_shot_idxs = []
    for idx, data_item in enumerate(tqdm(test_data)):
        zero_shot = False
        sparql = data_item["sparql"]
        pattern = r"ns:\S+"
        matches = re.findall(pattern, sparql)
        for match in matches:
            if match.startswith("ns:m.") or match.startswith("ns:g."):
                continue
            match = match[3:]
            if match not in train_rels_set:
                zero_shot = True
                break
        if zero_shot:
            test_zero_shot_idxs.append(idx)
        else:
            test_iid_idxs.append(idx)
    assert len(test_iid_idxs) + len(test_zero_shot_idxs) == len(test_data)
    with open(Config.ds_test_iid_idxs, "w") as f:  # 3433
        json.dump(test_iid_idxs, f)
    with open(Config.ds_test_zero_shot_idxs, "w") as f:  # 98
        json.dump(test_zero_shot_idxs, f)
    print(1)


def split_test_dep_by_iid_and_zero_shot():
    NSM_EGPSR_folder = f"/home/jjyu/IRQA/NSM_H/datasets/{Config.ds_tag}_EGPSR"
    test_dep_iid_idxs_file = os.path.join(NSM_EGPSR_folder, "test_iid.dep")
    test_dep_zero_shot_idxs_file = os.path.join(NSM_EGPSR_folder, "test_zero_shot.dep")
    if os.path.exists(test_dep_iid_idxs_file):
        os.remove(test_dep_iid_idxs_file)
    if os.path.exists(test_dep_zero_shot_idxs_file):
        os.remove(test_dep_zero_shot_idxs_file)
    with open(Config.ds_test_iid_idxs, "r") as f:
        test_iid_idxs = set(json.load(f))
    with open(Config.ds_test_zero_shot_idxs, "r") as f:
        test_zero_shot_idxs = set(json.load(f))
    test_dep = []
    with open(os.path.join(NSM_EGPSR_folder, "test.dep"), "r") as f:
        for line in f:
            test_dep.append(line)
    test_dep_iid = [item for idx, item in enumerate(test_dep) if idx in test_iid_idxs]
    test_dep_zero_shot = [
        item for idx, item in enumerate(test_dep) if idx in test_zero_shot_idxs
    ]
    assert len(test_dep_iid) == len(test_iid_idxs)
    assert len(test_dep_zero_shot) == len(test_dep_zero_shot)
    assert len(test_dep_iid) + len(test_dep_zero_shot) == len(test_dep)
    with open(test_dep_iid_idxs_file, "w") as f:
        for line in test_dep_iid:
            f.write(line)
    with open(test_dep_zero_shot_idxs_file, "w") as f:
        for line in test_dep_zero_shot:
            f.write(line)


def generate_webqsp_questionId2sparql():
    dest_file = "data/dataset/WebQSP/WebQSP_id2sparql.json"
    if os.path.exists(dest_file):
        print(f"{dest_file} already exists!")
        return
    id2sparql = {}
    with open("data/dataset/WebQSP/WebQSP.train.json", "r") as f:
        original_train = json.load(f)
        original_train = original_train["Questions"]
    with open("data/dataset/WebQSP/WebQSP.test.json", "r") as f:
        original_test = json.load(f)
        original_test = original_test["Questions"]
    for idx, item in enumerate(tqdm(original_train)):
        id2sparql[item["QuestionId"]] = item["Parses"][0]["Sparql"]
    for idx, item in enumerate(tqdm(original_test)):
        id2sparql[item["QuestionId"]] = item["Parses"][0]["Sparql"]
    with open(dest_file, "w") as f:
        json.dump(id2sparql, f)


def generate_webqsp_relations_in_train_set():
    if os.path.exists(Config.ds_train_relations):
        print(f"{Config.ds_train_relations} already exists!")
        return
    with open("data/dataset/WebQSP/WebQSP_id2sparql.json", "r") as f:
        id2sparql = json.load(f)
    train_data = []
    with open(Config.ds_train, "r") as f:
        for line in f:
            train_data.append(json.loads(line))
    train_rels_set = set()
    for idx, data_item in enumerate(tqdm(train_data)):
        sparql = id2sparql[data_item["id"]]
        pattern = r"ns:\S+"
        matches = re.findall(pattern, sparql)
        for match in matches:
            if match.startswith("ns:m.") or match.startswith("ns:g."):
                continue
            match = match[3:]
            train_rels_set.add(match)
    with open(Config.ds_train_relations, "w") as f:
        json.dump(list(train_rels_set), f)
    print(1)
    pass


def split_webqsp_test_by_iid_and_zero_shot():
    if os.path.exists(Config.ds_test_iid_idxs) and os.path.exists(
        Config.ds_test_zero_shot_idxs
    ):
        print(f"{Config.ds_test_iid_idxs} and {Config.ds_test_zero_shot_idxs} exist!")
        return
    test_data = []
    with open(Config.ds_test, "r") as f:
        for line in f:
            test_data.append(json.loads(line))
    with open(Config.ds_train_relations, "r") as f:
        train_rels_set = set(json.load(f))
    with open("data/dataset/WebQSP/WebQSP_id2sparql.json", "r") as f:
        id2sparql = json.load(f)
    test_iid_idxs = []
    test_zero_shot_idxs = []
    for idx, data_item in enumerate(tqdm(test_data)):
        zero_shot = False
        sparql = id2sparql[data_item["id"]]
        pattern = r"ns:\S+"
        matches = re.findall(pattern, sparql)
        for match in matches:
            if match.startswith("ns:m.") or match.startswith("ns:g."):
                continue
            match = match[3:]
            if match not in train_rels_set:
                zero_shot = True
                break
        if zero_shot:
            test_zero_shot_idxs.append(idx)
        else:
            test_iid_idxs.append(idx)
    assert len(test_iid_idxs) + len(test_zero_shot_idxs) == len(test_data)
    with open(Config.ds_test_iid_idxs, "w") as f:  # 3433
        json.dump(test_iid_idxs, f)
    with open(Config.ds_test_zero_shot_idxs, "w") as f:  # 98
        json.dump(test_zero_shot_idxs, f)
    print(1)


def get_mids_from_sparql(sparql):
    mids = set()
    re_expr = "ns:m\\.[0-9a-zA-Z_]+"
    pattern = re.compile(re_expr)
    temp_res = pattern.findall(sparql)
    for item in temp_res:
        mids.add(item)
    re_expr = "ns:g\\.[0-9a-zA-Z_]+"
    pattern = re.compile(re_expr)
    temp_res = pattern.findall(sparql)
    for item in temp_res:
        mids.add(item)
    return mids


def diff_topic_entities():
    test_items = load_ds_items(
        "data/dataset/CWQ/ComplexWebQuestions_test.json",
        "data/dataset/CWQ/CWQ_full_with_int_id.jsonl",
    )
    for idx, data_item in enumerate(test_items):
        topic_ents_in_sparql = get_mids_from_sparql(data_item.lf)
        topic_ents_sr = set(data_item.topic_ents)
        if "State" in data_item.lf:
            print(1)
        if (
            len(topic_ents_in_sparql - topic_ents_sr) > 0
            or len(topic_ents_sr - topic_ents_in_sparql) > 0
        ):
            print(1)
    print(1)


if __name__ == "__main__":
    # if Config.ds_tag == "CWQ":
    #     generate_cwq_relations_in_train_set()
    #     split_cwq_test_by_iid_and_zero_shot()
    #     split_test_dep_by_iid_and_zero_shot()
    # elif Config.ds_tag == "WebQSP":
    #     generate_webqsp_questionId2sparql()
    #     generate_webqsp_relations_in_train_set()
    #     split_webqsp_test_by_iid_and_zero_shot()
    #     split_test_dep_by_iid_and_zero_shot()
    diff_topic_entities()
    pass
