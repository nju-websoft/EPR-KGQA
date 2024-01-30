import json


def analynaze_generated_eps():
    dev_ep_list = []
    with open(
        "/home/jjyu/IRQA/data/CWQ/ep_retrieval/dev_top200_ap_ep_rank_td.jsonl", "r"
    ) as f:
        for line in f:
            dev_ep_list.append(json.loads(line))
    pos_max = 0
    neg_max = 0
    pos_avg = 0
    neg_avg = 0
    pos_max_item = None
    neg_max_item = None
    for item in dev_ep_list:
        pos_avg += len(item["positive"])
        neg_avg += len(item["negative"])
        if len(item["positive"]) > pos_max:
            pos_max = len(item["positive"])
            pos_max_item = item
        if len(item["negative"]) > neg_max:
            neg_max = len(item["negative"])
            neg_max_item = item
    pos_avg = pos_avg / len(dev_ep_list)
    neg_avg = neg_avg / len(dev_ep_list)
