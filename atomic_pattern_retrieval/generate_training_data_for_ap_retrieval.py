# Stardard Libraries
import csv
import json
import os
import random

# Third party libraries
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Self-defined Modules
from config import Config
from my_utils.io_utils import read_json, write_json


def create_freebase_rr_aps():
    dest_file = Config.cache_rr_aps
    if os.path.exists(dest_file):
        print(f'{dest_file} already exists!')
        return
    rel_conn_fb = {}
    with open(Config.cache_rel_conn) as f:
        for line in f:
            temp = json.loads(line)
            rel_conn_fb[temp["id"]] = temp["rel_conn"]
    if os.path.exists("data/cache/rr_aps_tag_dict.json"):
        rr_aps_tag_dict = read_json("data/cache/rr_aps_tag_dict.json")
    else:
        rr_aps_tag_dict = {"S-S": set(), "O-O": set(), "S-O": set()}
        for rel in tqdm(rel_conn_fb):
            assert rel.startswith('ns:')
            conn_dict = rel_conn_fb[rel]
            for tag in conn_dict:
                rel_list = conn_dict[tag]
                for another_rel in rel_list:
                    assert another_rel.startswith('ns:')
                    v1 = '\t'.join([rel[3:], another_rel[3:]])
                    v2 = '\t'.join([another_rel[3:], rel[3:]])
                    if tag == 'S-S' and v1 not in rr_aps_tag_dict['S-S'] and v2 not in rr_aps_tag_dict['S-S']:
                        rr_aps_tag_dict['S-S'].add(v1)
                    elif tag == 'S-O' and v1 not in rr_aps_tag_dict['S-O'] and v2 not in rr_aps_tag_dict['S-O']:
                        rr_aps_tag_dict['S-O'].add(v1)
                    elif tag == 'O-S' and v1 not in rr_aps_tag_dict['S-O'] and v2 not in rr_aps_tag_dict['S-O']:
                        rr_aps_tag_dict['S-O'].add(v2)
                    elif tag == 'O-O' and v1 not in rr_aps_tag_dict['O-O'] and v2 not in rr_aps_tag_dict['O-O']:
                        rr_aps_tag_dict['O-O'].add(v1)
        rr_aps_tag_dict['S-S'] = list(rr_aps_tag_dict['S-S'])  # 760797
        rr_aps_tag_dict['S-O'] = list(rr_aps_tag_dict['S-O'])  # 1131831
        rr_aps_tag_dict['O-O'] = list(rr_aps_tag_dict['O-O'])  # 511338
        write_json(rr_aps_tag_dict, "data/cache/rr_aps_tag_dict.json")
    rel_info_fb = read_json(Config.cache_rel_info)
    rr_aps_fb = []
    for tag in rr_aps_tag_dict:
        rr_aps = rr_aps_tag_dict[tag]
        for rr_ap in tqdm(rr_aps):
            r1, r2 = rr_ap.split('\t')
            if 'ns:' + r1 not in rel_info_fb:
                continue
            if 'ns:' + r2 not in rel_info_fb:
                continue
            rr_aps_fb.append(r1 + ' ' + tag + ' ' + r2)
    print('nums of RR-APs:', len(rr_aps_fb))
    write_json(rr_aps_fb, dest_file) # 2366590


def create_rr_aps_forward_and_reverse_dict(dest_file="data/cache/rr_aps_forward_reverse_dict.json", 
                                           rr_aps_tag_dict_file="data/cache/rr_aps_tag_dict.json"):
    if os.path.exists(dest_file):
        print(f'{dest_file} already exists!')
        return
    rel_info_fb = read_json(Config.cache_rel_info)
    dest_dict = {}
    rr_aps_tag_dict = read_json(rr_aps_tag_dict_file)
    for tag in rr_aps_tag_dict:
        rr_aps = rr_aps_tag_dict[tag]
        for rr_ap in tqdm(rr_aps):
            r1, r2 = rr_ap.split('\t')
            if 'ns:' + r1 not in rel_info_fb:
                continue
            if 'ns:' + r2 not in rel_info_fb:
                continue
            if r1 not in dest_dict:
                dest_dict[r1] = {"forward": {"S-S": [], "S-O": [], "O-O": []},
                                        "reverse": {"S-S": [], "S-O": [], "O-O": []}}
            if r2 not in dest_dict:
                dest_dict[r2] = {"forward": {"S-S": [], "S-O": [], "O-O": []},
                                        "reverse": {"S-S": [], "S-O": [], "O-O": []}}
            dest_dict[r1]["forward"][tag].append(r2)
            dest_dict[r2]["reverse"][tag].append(r1)
    write_json(dest_dict, dest_file)
    
    
def create_distantly_supervised_refined_relations(dest_file, split_file):
    if os.path.exists(dest_file):
        print(f'{dest_file} already exists!')
        return
    dest_data = []
    positive_rr_aps = read_json(split_file)
    rr_aps_fb = set(read_json(Config.cache_rr_aps))
    for idx, data_item in enumerate(tqdm(positive_rr_aps)):
        positive_rr_aps = data_item['positive']
        valid_positive_rr_aps = []
        # rr_ap format: f"p1:{p1[3::]}, p2:{p2[3::]}, tag:{tag}"
        for rr_ap in positive_rr_aps:
            r1, r2, tag = rr_ap.split(', ')
            assert r1.startswith('p1:')
            r1 = r1[3:]
            assert r2.startswith('p2:')
            r2 = r2[3:]
            assert tag.startswith('tag:')
            tag = tag[4:]
            if tag == 'O-S':
                tag = 'S-O'
                r1, r2 = r2, r1
            new_rr_ap = ' '.join([r1, tag, r2])
            if new_rr_ap not in rr_aps_fb:
                continue
            valid_positive_rr_aps.append(new_rr_ap)
        dest_item = {
            'id': data_item['id'],
            'split-id': data_item['split_id'],
            'question': data_item['question'],
            'distantly_supervised_rr_aps': valid_positive_rr_aps,
        }
        dest_data.append(dest_item)
    write_json(dest_data, dest_file)


def explore_max_length():
    rr_aps_fb = read_json(Config.cache_rr_aps)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    max_len = 0  # 95, with question considered 42
    for rr_ap in tqdm(rr_aps_fb):
        rr_ap_tokens = tokenizer.tokenize(rr_ap)
        max_len = max(max_len, len(rr_ap_tokens) + 2)
    print(max_len)  # 95
    print(1)


def sample_biencoder_data_hard_negative_random_negative(dest_file, split_file, sample_size=20):
    if os.path.exists(dest_file):
        print(f'{dest_file} already exists!')
        return
    data = read_json(split_file)
    rr_aps_fb = set(read_json(Config.cache_rr_aps))
    forward_and_reverse_dict = read_json("data/cache/rr_aps_forward_reverse_dict.json")
    samples = []
    tag_set = {'S-S', 'S-O', 'O-O'}
    for idx, data_item in enumerate(tqdm(data)):
        question = data_item['question']
        lower_question = question.lower()
        positive_rr_aps = data_item['distantly_supervised_rr_aps']
        all_diff_rr_aps = list(rr_aps_fb - set(positive_rr_aps))
        all_random_negative_rr_aps = set(random.sample(all_diff_rr_aps, 3 * sample_size * len(positive_rr_aps)))
        for pos_rr_ap in positive_rr_aps:
            sample = []
            sample.append([lower_question, pos_rr_ap, '1'])
            negative_rr_aps = set()
            r1, tag, r2 = pos_rr_ap.split(' ')
            # r1 且 r2 都存在，但tag 不同
            all_tags_cases = set([' '.join([r1, some_tag, r2]) for some_tag in tag_set])
            diff_tag_cases = all_tags_cases - set(positive_rr_aps)
            diff_tag_cases = diff_tag_cases.intersection(rr_aps_fb)
            negative_rr_aps = negative_rr_aps.union(diff_tag_cases)
            # r1 或 r2 存在任意一种
            right_rels = forward_and_reverse_dict[r1]['forward'][tag]
            diff_right_rel_cases = set([' '.join([r1, tag, right_rel]) for right_rel in right_rels])
            left_rels = forward_and_reverse_dict[r2]['reverse'][tag]
            diff_left_rel_cases = set([' '.join([left_rel, tag, r2]) for left_rel in left_rels])
            diff_rel_cases = diff_right_rel_cases.union(diff_left_rel_cases)
            diff_rel_cases = diff_rel_cases - set(positive_rr_aps)
            diff_rel_cases = diff_rel_cases.intersection(rr_aps_fb)
            partial_diff_rel_cases = random.sample(list(diff_rel_cases),
                                                   min(len(diff_rel_cases), sample_size // 2 - len(diff_tag_cases)))
            negative_rr_aps = negative_rr_aps.union(partial_diff_rel_cases)
            # random negative
            diff_rr_aps = all_random_negative_rr_aps - negative_rr_aps
            random_negative = random.sample(diff_rr_aps, sample_size - len(negative_rr_aps) - 1)
            negative_rr_aps = negative_rr_aps.union(set(random_negative))
            assert len(negative_rr_aps) == sample_size - 1
            for n_rel in negative_rr_aps:
                sample.append([lower_question, n_rel, '0'])
            random.shuffle(sample)
            samples.extend(sample)
    with open(dest_file, 'w') as f:
        header = ['id', 'question', 'relation', 'label']
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        idx = 0
        for line in samples:
            writer.writerow([str(idx)] + line)
            idx += 1


def create_feature_cache_for_training_hard_negative_random_negative(dest_file, split_file, sample_size=20,
                                                                    max_len=95):
    if os.path.exists(dest_file):
        print(f'{dest_file} already exists!')
        return
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    data = pd.read_csv(split_file, delimiter='\t', dtype={"id": int, "question": str, "relation": str, 'label': int})
    samples = len(data) // sample_size
    dest_data = []
    for index in tqdm(range(samples)):
        start = sample_size * index
        end = min(sample_size * (index + 1), len(data))
        question = str(data.loc[start, 'question'])
        relations = [str(data.loc[i, 'relation']) for i in range(start, end)]
        golden_id = [i - start for i in range(start, end) if data.loc[i, 'label'] == 1]
        assert len(golden_id) == 1, print(start, end)
        encoded_question = tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )
        encoded_relations = [tokenizer(
            relation,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        ) for relation in relations]

        question_token_ids = encoded_question['input_ids'].squeeze(0)  # tensor of token ids
        question_attn_masks = encoded_question['attention_mask'].squeeze(
            0)  # binary tensor with "0" for padded values and "1" for the other values
        question_token_type_ids = encoded_question['token_type_ids'].squeeze(
            0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        relations_token_ids = torch.cat([encoded_relation['input_ids'] for encoded_relation in encoded_relations], 0)
        relations_attn_masks = torch.cat([encoded_relation['attention_mask'] for encoded_relation in encoded_relations],
                                         0)
        relations_token_type_ids = torch.cat(
            [encoded_relation['token_type_ids'] for encoded_relation in encoded_relations], 0)
        dest_item = {
            'question_tokens_ids': question_token_ids,
            'question_attn_masks': question_attn_masks,
            'question_token_type_ids': question_token_type_ids,
            'relation_tokens_ids': relations_token_ids,
            'relation_attn_masks': relations_attn_masks,
            'relation_token_type_ids': relations_token_type_ids,
            'golden_id': golden_id[0]
        }
        dest_data.append(dest_item)
    torch.save(dest_data, dest_file)


if __name__ == '__main__':
    create_freebase_rr_aps()
    create_rr_aps_forward_and_reverse_dict()

    # explore_max_length()
    sample_size = 20
    
    for split in ['dev', 'train']:
        create_distantly_supervised_refined_relations(
            dest_file=f"data/{Config.ds_tag}/ap_retrieval/training_data/{Config.ds_tag}_{split}_distantly_supervised_data.json",
            split_file=f"data/{Config.ds_tag}/ap_retrieval/training_data/{Config.ds_tag}_{split}_positive_rr_aps.json")
        sample_biencoder_data_hard_negative_random_negative(
            dest_file=f"data/{Config.ds_tag}/ap_retrieval/training_data/{Config.ds_tag}_{split}_training_data_sample_size_{sample_size}.tsv", 
            split_file=f"data/{Config.ds_tag}/ap_retrieval/training_data/{Config.ds_tag}_{split}_distantly_supervised_data.json",
            sample_size=sample_size)
        create_feature_cache_for_training_hard_negative_random_negative(
            dest_file=f"data/{Config.ds_tag}/ap_retrieval/training_data/{Config.ds_tag}_{split}_feature_cache_sample_size_{sample_size}",
            split_file=f"data/{Config.ds_tag}/ap_retrieval/training_data/{Config.ds_tag}_{split}_training_data_sample_size_{sample_size}.tsv",
            sample_size=sample_size)
        