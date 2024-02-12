# Stardard Libraries
import os

# Third party libraries
from tqdm import tqdm

# Self-defined Modules
from config import Config
from my_utils.io_utils import read_json, append_line, append_jsonl
from my_utils.logger import Logger
from my_utils.rel_base import relBase

logger = Logger.get_logger("Convert2NSM", True)


def collect_covered_ent_rel(retrieved_subgs):
    ents = set()
    rels = set()
    for subg_info in tqdm(retrieved_subgs):
        for tripstr in subg_info["subg"]:
            trips = tripstr.split(" ")
            if trips[0].startswith('ns:'):
                ents.add(trips[0][3::])
            else:
                ents.add(trips[0])
            if trips[2].startswith('ns:'):
                ents.add(trips[2][3::])
            else:
                ents.add(trips[2])
            assert trips[1].startswith('ns:')
            rels.add(trips[1][3::])
            if relBase.get_reverse(trips[1]):
                rels.add(relBase.get_reverse(trips[1])[3::])
        for topic in subg_info["topics"]:
            if topic.startswith('ns:'):
                ents.add(topic[3::])
            else:
                ents.add(topic)
        for ans in subg_info["answers"]:
            if ans.startswith('ns:'):
                ents.add(ans[3::])
            else:
                ents.add(ans)
    return ents, rels


def load_itemid_map(item_file):
    item2id = dict()
    if not os.path.exists(item_file):
        return item2id
    for line in open(item_file):
        item = line.strip()
        item2id[item] = len(item2id)
    return item2id


def trans_to_nsm_format(ent2id, rel2id, ours_subgs, reformated_subgs_f):
    reformated_subgs = []
    for subg_info in ours_subgs:
        topic_ents = [ent2id[topic[3::]] for topic in subg_info["topics"]]
        subg_ents = []
        subg_tuples = []
        for tripstr in subg_info["subg"]:
            trips = tripstr.split(" ")
            if trips[0].startswith('ns:'):
                subj = trips[0][3::]
            else:
                subj = trips[0]
            assert  trips[1].startswith('ns:')
            rel = trips[1][3::]
            if relBase.get_reverse(trips[1]):
                rel_rev = relBase.get_reverse(trips[1])[3::]
            else:
                rel_rev = None
            if trips[2].startswith('ns:'):
                obj = trips[2][3::]
            else:
                obj = trips[2]
            subg_ents.append(ent2id[subj])
            subg_ents.append(ent2id[obj])
            subg_tuples.append([ent2id[subj], rel2id[rel], ent2id[obj]])
            if rel_rev:
                subg_tuples.append([ent2id[obj], rel2id[rel_rev], ent2id[subj]])
        subg_ents = list(set(subg_ents))
        nsm_item = dict()
        nsm_item["id"] = subg_info["id"]
        nsm_item["answers"] = []
        for ans in subg_info["answers"]:
            if ans.startswith('ns:'):
                nsm_item["answers"].append({"kb_id": ans[3::], "text": None})
            else:
                nsm_item["answers"].append({"kb_id": ans, "text": None})
        nsm_item["question"] = subg_info["question"]
        nsm_item["entities"] = topic_ents
        nsm_item["subgraph"] = dict()
        nsm_item["subgraph"]["entities"] = subg_ents
        nsm_item["subgraph"]["tuples"] = subg_tuples
        reformated_subgs.append(nsm_item)
    if os.path.exists(reformated_subgs_f):
        os.remove(reformated_subgs_f)
    for subg_info in reformated_subgs:
        append_jsonl(subg_info, reformated_subgs_f)


def update_entity_and_relation_file_by_test_file(topk_range=[80, 100], step=20):
    entity_file = Config.nsm_entities
    relation_file = Config.nsm_relations
    ent2id = load_itemid_map(entity_file)
    rel2id = load_itemid_map(relation_file)

    print(">>> collect used entities & relations")
    used_ents = set()
    used_rels = set()

    # collect entities & relations from induced subgraph (test ap=[20:200], ep=1)
    topks = range(topk_range[0], topk_range[1] + 1, step)
    for topk in topks:
        Config.ap_topk = topk
        test_ents, test_rels = collect_covered_ent_rel(
            read_json(Config.induced_subg_f("test"))
        )
        used_ents |= test_ents
        used_rels |= test_rels
    for ent in used_ents:
        if ent not in ent2id:
            append_line(ent, entity_file)
    for rel in used_rels:
        if rel not in rel2id:
            append_line(rel, relation_file)


def update_entity_and_relation_file_by_train_and_dev():
    entity_file = Config.nsm_entities
    relation_file = Config.nsm_relations
    ent2id = load_itemid_map(entity_file)
    rel2id = load_itemid_map(relation_file)

    print(">>> collect used entities & relations")
    # collect entities & relations from induced subgraph (train ap=100, ep=3)
    train_topk_subg = read_json(Config.ans_rank_td_f("train"))
    dev_topk_subg = read_json(Config.ans_rank_td_f("dev"))

    train_ents, train_rels = collect_covered_ent_rel(train_topk_subg)
    dev_ents, dev_rels = collect_covered_ent_rel(dev_topk_subg)
    used_ents = train_ents | dev_ents
    used_rels = train_rels | dev_rels

    for ent in used_ents:
        if ent not in ent2id:
            append_line(ent, entity_file)
    for rel in used_rels:
        if rel not in rel2id:
            append_line(rel, relation_file)

    print(">>> reformat ours subgs")
    ent2id = load_itemid_map(entity_file)
    rel2id = load_itemid_map(relation_file)
    trans_to_nsm_format(ent2id, rel2id, train_topk_subg, Config.nsm_input_f("train"))
    trans_to_nsm_format(ent2id, rel2id, dev_topk_subg, Config.nsm_input_f("dev"))


def generate_test_file(topk=100):
    entity_file = Config.nsm_entities
    relation_file = Config.nsm_relations
    ent2id = load_itemid_map(entity_file)
    rel2id = load_itemid_map(relation_file)
    Config.ap_topk = topk
    test_topk_subg = read_json(Config.induced_subg_f("test"))
    trans_to_nsm_format(ent2id, rel2id, test_topk_subg, Config.nsm_input_f("test"))


if __name__ == "__main__":
    if not os.path.exists(Config.nsm_dir):
        os.makedirs(Config.nsm_dir)
    Config.ep_topk = 1
    update_entity_and_relation_file_by_test_file(topk_range=[80, 100], step=20)
    Config.ep_topk = 3
    Config.ap_topk = 100
    update_entity_and_relation_file_by_train_and_dev()
    Config.ep_topk = 1
    # only topk in topk_range is allowed, otherwise there may be 
    # entities or relationships that have not been seen before.
    generate_test_file(topk=100)
