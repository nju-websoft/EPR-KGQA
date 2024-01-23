import json
import pandas as pd
from tqdm import tqdm
tqdm.monitor_interval = 0

from config import Config

def analyze_prop_with_datatype_WikibaseItem():
    df = pd.read_csv("/home/jjyu/IRQA/data/cache/wikidata/wikidata_tables.csv")
    df_wi = df.loc[df['Data type[1]'] == 'WI', ['ID']]
    df_wi_list = df_wi.values.tolist()
    props_wi = [item[0] for item in df_wi_list]
    final_cache = {}
    forward_cache = {}
    with open('/home/jjyu/IRQA/data/cache/wikidata/wikidata_props_wi_conn_cache.jsonl', 'r') as f:
        for line in f:
            temp = json.loads(line)
            forward_cache[temp['id']] = temp['pred_conn']
    reverse_cache = {}
    with open('/home/jjyu/IRQA/data/cache/wikidata/wikidata_props_wi_conn_cache_reverse.jsonl', 'r') as f:
        for line in f:
            temp = json.loads(line)
            reverse_cache[temp['id']] = temp['pred_conn']
    for prop in props_wi:
        if prop in forward_cache and prop in reverse_cache:
            assert forward_cache[prop] == reverse_cache[prop]
            final_cache[prop] = forward_cache[prop]
        elif prop in forward_cache:
            final_cache[prop] = forward_cache[prop]
        elif prop in reverse_cache:
            final_cache[prop] = reverse_cache[prop]
        else:
            print('Error!')
    for id in final_cache:
        with open('/home/jjyu/IRQA/data/cache/wikidata/wikidata_props_WikibaseItem_conn_cache.jsonl', 'a') as wf:
            wf.write(json.dumps({"id": id, "pred_conn": final_cache[id]}) + '\n')
    print(1)


def generate_RR_APs_for_Wikidata():
    prop_conn_wiki = {}
    with open('/home/jjyu/IRQA/data/cache/wikidata/wikidata_props_WikibaseItem_conn_cache.jsonl', 'r') as f:
        for line in f:
            temp = json.loads(line)
            prop_conn_wiki[temp["id"]] = temp["pred_conn"]
    RR_APs = {"S-S": set(), "O-O": set(), "S-O": set()}
    for prop in tqdm(prop_conn_wiki):
        conn_dict = prop_conn_wiki[prop]
        for tag in conn_dict:
            prop_list = conn_dict[tag]
            for another_prop in prop_list:
                v1 = '\t'.join([prop, another_prop])
                v2 = '\t'.join([another_prop, prop])
                if tag == 'S-S' and v1 not in RR_APs['S-S'] and v2 not in RR_APs['S-S']:
                    RR_APs['S-S'].add(v1)
                elif tag == 'S-O' and v1 not in RR_APs['S-O'] and v2 not in RR_APs['S-O']:
                    RR_APs['S-O'].add(v1)
                elif tag == 'O-S' and v1 not in RR_APs['S-O'] and v2 not in RR_APs['S-O']:
                    RR_APs['S-O'].add(v2)
                elif tag == 'O-O' and v1 not in RR_APs['O-O'] and v2 not in RR_APs['O-O']:
                    RR_APs['O-O'].add(v1)
    RR_APs['S-S'] = list(RR_APs['S-S'])
    RR_APs['S-O'] = list(RR_APs['S-O'])
    RR_APs['O-O'] = list(RR_APs['O-O'])
    total_RR_APs = len(RR_APs['S-S']) + len(RR_APs['S-O']) + len(RR_APs['O-O'])
    print(1)


def generate_wdt_RR_APs_for_Wikidata():
    prop_conn_wiki = {}
    with open('/home/jjyu/IRQA/data/cache/wikidata/wikidata_props_WikibaseItem_conn_cache.jsonl', 'r') as f:
        for line in f:
            temp = json.loads(line)
            prop_conn_wiki[temp["id"]] = temp["pred_conn"]
    df = pd.read_csv("/home/jjyu/IRQA/data/cache/wikidata/wikidata_tables.csv")
    df_wi = df.loc[df['Data type[1]'] == 'WI', ['ID']]
    df_wi_list = df_wi.values.tolist()
    props_wi = [item[0] for item in df_wi_list]
    props_wi_set = set(props_wi)
    RR_APs = {"S-S": set(), "O-O": set(), "S-O": set()}
    for prop in tqdm(prop_conn_wiki):
        conn_dict = prop_conn_wiki[prop]
        for tag in conn_dict:
            prop_list = conn_dict[tag]
            for another_prop in prop_list:
                if another_prop not in props_wi_set:
                    continue
                v1 = '\t'.join([prop, another_prop])
                v2 = '\t'.join([another_prop, prop])
                if tag == 'S-S' and v1 not in RR_APs['S-S'] and v2 not in RR_APs['S-S']:
                    RR_APs['S-S'].add(v1)
                elif tag == 'S-O' and v1 not in RR_APs['S-O'] and v2 not in RR_APs['S-O']:
                    RR_APs['S-O'].add(v1)
                elif tag == 'O-S' and v1 not in RR_APs['S-O'] and v2 not in RR_APs['S-O']:
                    RR_APs['S-O'].add(v2)
                elif tag == 'O-O' and v1 not in RR_APs['O-O'] and v2 not in RR_APs['O-O']:
                    RR_APs['O-O'].add(v1)
    RR_APs['S-S'] = list(RR_APs['S-S'])
    RR_APs['S-O'] = list(RR_APs['S-O'])
    RR_APs['O-O'] = list(RR_APs['O-O'])
    total_RR_APs = len(RR_APs['S-S']) + len(RR_APs['S-O']) + len(RR_APs['O-O'])
    RR_APs_list = []
    for tag in RR_APs:
        snippets = RR_APs[tag]
        for snippet in tqdm(snippets):
            r1, r2 = snippet.split('\t')
            RR_APs_list.append(r1 + ' ' + tag + ' ' + r2)
    print('nums of RR-APs:', len(RR_APs_list))
    # with open('/home/jjyu/IRQA/data/cache/wikidata/wikidata_props_WikibaseItem_RR_APs_374647.json', 'w') as wf:
    #     json.dump(RR_APs_list, wf, indent=2)
    print(1)


def calculate_RR_APs_for_Freebase():
    with open('/home/jjyu/IR4WG/IRQA/data/training_data/cwq_snippets_biencoder/freebase_snippets_list.json', 'r') as f:
        fb_rr_aps = json.load(f)
    with open(Config.cache_pred_info, 'r') as f:
        pred_info_fb = json.load(f)
    valid_rr_aps = []
    for idx, rr_ap in enumerate(tqdm(fb_rr_aps)):
        r1, tag, r2 = rr_ap.split(' ')
        if r1.startswith('user.') or r1.startswith('base.') or r2.startswith('user.') or r2.startswith('base.'):
            continue
        valid_rr_aps.append(rr_ap)
    print(1)


if __name__ == "__main__":
    # analyze_prop_with_datatype_WikibaseItem()
    # generate_RR_APs_for_Wikidata()
    # generate_wdt_RR_APs_for_Wikidata()
    calculate_RR_APs_for_Freebase()
