# Stardard Libraries
import json
from multiprocessing import Process, Queue
import os
import random


random.seed(1)


def read_jsonl_by_key(filepath: str, key: str = "id") -> dict:
    result = dict()
    if os.path.exists(filepath):
        with open(filepath) as f:
            for line in f:
                temp = json.loads(line)
                result[temp[key]] = temp
    return result


def read_json_list_by_key(filepath: str, key: str = "id") -> dict:
    raw_list = read_json(filepath)
    result = dict()
    for item in raw_list:
        result[item[key]] = item
    return result


def append_jsonl(info: dict, filename: str):
    line = json.dumps(info, ensure_ascii=False)
    with open(filename, "a") as f:
        f.write(line + "\n")


def append_line(line: str, filename: str):
    with open(filename, "a") as f:
        f.write(line + "\n")


def read_json(json_file: str):
    res = None
    with open(json_file) as f:
        res = json.load(f)
    return res


def write_json(out_obj, outfile: str):
    with open(outfile, "w") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)


def run_multiprocess(target_func, args_list, items_to_deal, worker_num):
    processes = []
    queue = Queue(maxsize=100000)
    for i in range(worker_num):
        process = Process(target=target_func, args=tuple(args_list + [i, queue]))
        processes.append(process)

    for p in processes:
        p.start()

    for item in items_to_deal:
        queue.put(item)

    for i in range(worker_num):
        queue.put(None)

    for p in processes:
        p.join()


def find_in_list(entry, elist):
    for item in elist:
        if entry == item:
            return True
    return False


def calculate_PRF1(goldAnswerList, predAnswerList, hard=True):
    goldAnswerList = list(set(goldAnswerList))
    predAnswerList = list(set(predAnswerList))
    if len(goldAnswerList) == 0:
        if hard:
            return [0.0, 0.0, 0.0]
        elif len(predAnswerList) == 0:
            return [
                1.0,
                1.0,
                1.0,
            ]  # consider it 'correct' when there is no labeled answer, and also no predicted answer
        else:
            return [
                0.0,
                1.0,
                0.0,
            ]  # precision=0 and recall=1 when there is no labeled answer, but has some predicted answer(s)
    elif len(predAnswerList) == 0:
        if hard:
            return [0.0, 0.0, 0.0]
        else:
            return [
                1.0,
                0.0,
                0.0,
            ]  # precision=1 and recall=0 when there is labeled answer(s), but no predicted answer
    else:
        glist = goldAnswerList
        plist = predAnswerList

        tp = 1e-40  # numerical trick
        fp = 0.0
        fn = 0.0

        for gentry in glist:
            if find_in_list(gentry, plist):
                tp += 1
            else:
                fn += 1
        for pentry in plist:
            if not find_in_list(pentry, glist):
                fp += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f1 = (2 * precision * recall) / (precision + recall)
        return [precision, recall, f1]


def calculate_hits1(goldAnswerList, predAnswerList, random_choose=True) -> float:
    # 若预测结果有多个，取第一个
    ans_set = set(goldAnswerList)
    if len(goldAnswerList) == 0 or len(predAnswerList) == 0:
        return 0.0
    else:
        if random_choose:
            select_one = random.choice(predAnswerList)
        else:
            select_one = predAnswerList[0]
        if select_one in ans_set:
            return 1.0
        else:
            return 0.0
