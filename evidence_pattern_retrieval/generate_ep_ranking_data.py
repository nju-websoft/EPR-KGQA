# Stardard Libraries
import json
import math
import os
import random
import re
import time

# Third party libraries
from SPARQLWrapper import SPARQLWrapper, JSON
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Self-defined Modules
from BERT_Ranker.BertRanker import RankingFeature
from config import Config
from my_utils.data_item import load_ds_items
from my_utils.io_utils import read_json, write_json


sparql_wrapper = SPARQLWrapper('http://127.0.0.1:1111/sparql')
sparql_wrapper.setReturnFormat(JSON)


def get_entity_name_from_mid(mid):
    sparql_query = """PREFIX ns: <http://rdf.freebase.com/ns/>
        select distinct ?name
        where { """ + 'ns:{0}'.format(mid) + """ ns:type.object.name ?name.
            FILTER(LANG(?name) = "en")
        }"""
    sparql_wrapper.setQuery(sparql_query)
    result = sparql_wrapper.query().convert()
    bindings = result['results']['bindings']
    if len(bindings) > 0:
        result = result['results']['bindings'][0]['name']['value']
    else:
        result = mid
    return result


def update_dict_by_entities(entity_name_dict, entities):
    for entity in entities:
        name = entity
        try:
            name = get_entity_name_from_mid(entity)
        except Exception as e:
            print(e)
            time.sleep(2)
            print('try again')
            try:
                name = get_entity_name_from_mid(entity)
            except Exception as e1:
                print(e1)
                print('error mid:', entity)
        if name != entity or (name == entity and not (name.startswith('m.') or name.startswith('g.'))):
            entity_name_dict[entity] = name
        else:
            entity_name_dict[entity] = '[CVT]'


def create_entity_name_dict_by_split(entity_name_dict, split):
    ds_items = load_ds_items(Config.ds_split_f(split))
    for idx, data_item in enumerate(tqdm(ds_items)):
        entities = []
        for entity in data_item.topic_ents:
            if entity.startswith('ns:'):
                entities.append(entity[3:])
            else:
                entities.append(entity)
        update_dict_by_entities(entity_name_dict, entities)


def create_entity_name_dict():
    if os.path.exists(entity_name_dict_file):
        print(f'{entity_name_dict_file} already exists!')
        return
    entity_name_dict = {}
    create_entity_name_dict_by_split(entity_name_dict, 'dev')
    create_entity_name_dict_by_split(entity_name_dict, 'test')
    create_entity_name_dict_by_split(entity_name_dict, 'train')
    write_json(entity_name_dict, entity_name_dict_file)
    print(1)


entity_name_dict_file = os.path.join(Config.ep_retrieval_dir, f'{Config.ds_tag}_entity_name_dict.json')
if not os.path.exists(entity_name_dict_file):
    create_entity_name_dict()
with open(entity_name_dict_file, 'r') as f:
    entity_name_dict = json.load(f)


def get_mids_from_ep(ep):
    mids = set()
    re_expr = 'ns:m\\.[0-9a-zA-Z_]+'
    pattern = re.compile(re_expr)
    temp_res = pattern.findall(ep)
    for item in temp_res:
        mids.add(item)
    re_expr = 'ns:g\\.[0-9a-zA-Z_]+'
    pattern = re.compile(re_expr)
    temp_res = pattern.findall(ep)
    for item in temp_res:
        mids.add(item)
    return list(mids)


def replace_mid_by_name(ep):
    mids = get_mids_from_ep(ep)
    for mid in mids:
        ep = ep.replace(mid, entity_name_dict[mid[3:]])
    return ep


def generate_data_for_training(split, sample_size):
    dest_file = os.path.join(
        Config.ep_retrieval_dir, 'training_data',
        f'{Config.ds_tag}_{split}_training_data_sample_size_{sample_size}.json')
    if os.path.exists(dest_file):
        print(f'{dest_file} already exists!')
        return
    if not os.path.exists(os.path.dirname(dest_file)):
        os.makedirs(os.path.dirname(dest_file))
    data = []
    with open(Config.ep_rank_td_f(split), 'r') as f:
        for line in f:
            data.append(json.loads(line))
    dest_data = []
    negative_samples = sample_size - 1
    for idx, data_item in enumerate(tqdm(data)):
        if len(data_item['positive_eps']) == 0:
            continue
        positive_eps = data_item['positive_eps']
        random.shuffle(positive_eps)
        negative_eps = data_item['negative_eps']
        random.shuffle(negative_eps)
        ranking_problems = []
        for serial_num, positive_ep in enumerate(positive_eps):
            raw_candidates = negative_eps[
                             serial_num * negative_samples: (serial_num + 1) * negative_samples]
            if len(raw_candidates) == 0:
                break
            raw_candidates.append(positive_ep)
            random.shuffle(raw_candidates)
            candidates = [replace_mid_by_name(candidate) for candidate in raw_candidates]
            ranking_problems.append({
                "problem_id": f'{split}-{idx}-{serial_num}',
                "positive_ep": replace_mid_by_name(positive_ep),
                "candidates": candidates
            })
        dest_data.append({
            'id': data_item['id'],
            'split_id': f'{split}-{idx}',
            'question': data_item['question'],
            'ranking_problems': ranking_problems
        })
    write_json(dest_data, dest_file)
    print(1)


def generate_feature_cache_for_training(split, sample_size, max_seq_length=256):
    dest_file = os.path.join(
        Config.ep_retrieval_dir, 'training_data',
        f'{Config.ds_tag}_{split}_feature_cache_for_training_sample_size_{sample_size}')
    if os.path.exists(dest_file):
        print(f'{dest_file} already exists!')
        return
    data = read_json(os.path.join(Config.ep_retrieval_dir, 'training_data',
                                 f'{Config.ds_tag}_{split}_training_data_sample_size_{sample_size}.json'))
    features = []
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    special_tokens_dict = {'additional_special_tokens': ['[CVT]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    max_len = 0  # dev 86 train 115
    for idx, data_item in enumerate(tqdm(data)):
        question = data_item['question'].lower()
        ranking_problems = data_item['ranking_problems']
        for ranking_problem in ranking_problems:
            candidates = ranking_problem['candidates']
            pid = ranking_problem['problem_id']
            positive_ep = ranking_problem['positive_ep']
            candidate_input_ids = []
            candidate_token_type_ids = []
            for candidate in candidates:
                ep = candidate.lower()
                c_encoded = tokenizer(question, ep, truncation=True, max_length=max_seq_length,
                                      return_token_type_ids=True)
                max_len = max(max_len, len(c_encoded['input_ids']))
                candidate_input_ids.append(c_encoded['input_ids'])
                candidate_token_type_ids.append(c_encoded['token_type_ids'])
            target_idx = next((i for (i, x) in enumerate(candidates) if x == positive_ep), -1)
            assert target_idx != -1
            features.append(
                RankingFeature(pid, candidate_input_ids, candidate_token_type_ids, target_idx))
    torch.save(features, dest_file)
    print(1)


def generate_data_for_prediction(split, sample_size, topk=100):
    dest_file = os.path.join(Config.ep_retrieval_dir, 'training_data',
                             f'{Config.ds_tag}_{split}_top{topk}_ap_candi_eps_for_prediction.json')
    if os.path.exists(dest_file):
        print(f'{dest_file} already exists!')
        return
    data = read_json(Config.candi_ep_f(split))
    dest_data = []
    for idx, data_item in enumerate(tqdm(data)):
        raw_candidates = data_item['candidate_eps']
        random.shuffle(raw_candidates)
        candidates = [[candidate, replace_mid_by_name(candidate)] for candidate in raw_candidates]
        num_problems = math.ceil(len(candidates) / sample_size)
        ranking_problems = []
        for serial_num in range(num_problems):
            ranking_problems.append({
                "problem_id": f'test-{idx}-{serial_num}',
                "candidates": candidates[serial_num * sample_size: (serial_num + 1) * sample_size]
            })
        dest_data.append({
            'id': data_item['id'],
            'split_id': f'test-{idx}',
            'question': data_item['question'],
            'ranking_problems': ranking_problems
        })
    write_json(dest_data, dest_file)
    print(1)


def generate_feature_cache_for_prediction(split, topk=100, max_seq_length=256):
    dest_file = os.path.join(Config.ep_retrieval_dir, 'training_data',
                             f'{Config.ds_tag}_{split}_top{topk}_ap_candi_eps_feature_cache_for_prediction')
    if os.path.exists(dest_file):
        print(f'{dest_file} already exists!')
        return
    data = read_json(os.path.join(Config.ep_retrieval_dir, 'training_data',
                                  f'{Config.ds_tag}_{split}_top{topk}_ap_candi_eps_for_prediction.json'))
    features = []
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    special_tokens_dict = {'additional_special_tokens': ['[CVT]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    for idx, data_item in enumerate(tqdm(data)):
        question = data_item['question'].lower()
        ranking_problems = data_item['ranking_problems']
        for ranking_problem in ranking_problems:
            candidates = [pair[1] for pair in ranking_problem['candidates']]
            pid = ranking_problem['problem_id']
            candidate_input_ids = []
            candidate_token_type_ids = []
            for candidate in candidates:
                ep = candidate.lower()
                c_encoded = tokenizer(question, ep, truncation=True, max_length=max_seq_length,
                                      return_token_type_ids=True)
                candidate_input_ids.append(c_encoded['input_ids'])
                candidate_token_type_ids.append(c_encoded['token_type_ids'])
            target_idx = -1
            features.append(
                RankingFeature(pid, candidate_input_ids, candidate_token_type_ids, target_idx))
    torch.save(features, dest_file)
    print(1)


if __name__ == '__main__':
    sample_size = Config.ep_rank_sample_size
    for split in ['dev', 'train']:
        generate_data_for_training(split=split, sample_size=sample_size)
        generate_feature_cache_for_training(split=split, sample_size=sample_size)
    topk=100
    Config.ap_topk = topk
    for split in ['test', 'dev', 'train']:
        generate_data_for_prediction(split=split, sample_size=sample_size, topk=topk)
        generate_feature_cache_for_prediction(split=split, topk=topk)
    split = 'test'
    topk = 80
    Config.ap_topk = topk
    generate_data_for_prediction(split=split, sample_size=sample_size, topk=topk)
    generate_feature_cache_for_prediction(split=split, topk=topk)
    print(1)
