import json

# WEBQSP_ENTITY_FILE = "datasets/webqsp/entities.txt"
# WEBQSP_SIMPLE_FILE = "datasets/webqsp/test_simple.json"
# WEBQSP_RESULT_FILE = "checkpoint/webqsp_report/student/webqsp_hybrid_student_test.info"
# WEBQSP_OUT_FILE = "WebQSPTestResult.json"

# CWQ_ENTITY_FILE = "datasets/CWQ/entities.txt"
# CWQ_SIMPLE_FILE = "datasets/CWQ/test_simple.json"
# CWQ_RESULT_FILE = "checkpoint/CWQ_report/student/CWQ_hybrid_student_test.info"
# CWQ_OUT_FILE = "CWQTestResult.json"

CWQ_ENTITY_FILE = "datasets/CWQ_EGPSR/entities.txt"
CWQ_SIMPLE_FILE = "datasets/CWQ_EGPSR/test_simple.json"
CWQ_RESULT_FILE = "checkpoint/pretrain/CWQ_nsm_test.info"
CWQ_OUT_FILE = "OursCWQTestResult.json"

WEBQSP_ENTITY_FILE = "datasets/WebQSP_EGPSR/entities.txt"
WEBQSP_SIMPLE_FILE = "datasets/WebQSP_EGPSR/test_simple.json"
WEBQSP_RESULT_FILE = "checkpoint/pretrain/WebQSP_nsm_test.info"
WEBQSP_OUT_FILE = "OursWebQSPTestResult.json"


def load_entity(filename):
    ents = []
    with open(filename, encoding="utf-8") as f_in:
        for line in f_in:
            ent = line.strip()
            ents.append(ent)
    return ents


def load_dataset(filename):
    dataset = []
    with open(filename, encoding="utf-8") as f_in:
        for line in f_in:
            temp = json.loads(line)
            dataset.append(
                {
                    "question": temp["question"],
                    "topicEnt": temp["entities"],
                    "goldAns": [d["kb_id"] for d in temp["answers"]],
                }
            )
    return dataset


def load_result(filename):
    results = []
    with open(filename, encoding="utf-8") as f_in:
        for line in f_in:
            temp = json.loads(line)
            results.append(
                {
                    "question": "",
                    "topicEnt": [],
                    "goldAns": [],
                    "candidate": temp["candidate"],
                    "P": temp["precison"],
                    "R": temp["recall"],
                    "f1": temp["f1"],
                    "hit": temp["hit"],
                }
            )
    return results


def integrate(entity, dataset, result, out):
    ents = load_entity(entity)
    testset = load_dataset(dataset)
    results = load_result(result)
    for i in range(len(results)):
        item = results[i]
        temp = testset[i]
        item["question"] = temp["question"]
        item["topicEnt"] = [ents[i] for i in temp["topicEnt"]]
        item["goldAns"] = temp["goldAns"]
        item["candidate"] = [[ents[l[0]], l[1]] for l in item["candidate"]]
    # with open(out, "w") as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)


integrate(WEBQSP_ENTITY_FILE, WEBQSP_SIMPLE_FILE, WEBQSP_RESULT_FILE, WEBQSP_OUT_FILE)
# integrate(CWQ_ENTITY_FILE,CWQ_SIMPLE_FILE,CWQ_RESULT_FILE,CWQ_OUT_FILE)

# { "tokenQ": "[CLS] what does jamaican people speak [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [
# PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [
# PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [
# PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]", "topicEnt": "m.03_r3", "goldAns": [ "m.01428y",
# "m.04ygk0" ], "bestEnt": "m.01428y", "mScore": 1.0, "candidate": [ [ "m.01428y", 0.9990565180778503 ],
# [ "m.04ygk0", 0.9990565180778503 ] ] },
