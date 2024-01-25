# Third party libraries
from tqdm.contrib import tzip

# Self-defined Modules
from my_utils.data_item import load_ds_items
from my_utils.freebase import FreebaseODBC
from my_utils.io_utils import read_json, append_jsonl
from preprocess.heuristic_path_search import forward_search


freebase = FreebaseODBC()

raw_nsm_results = read_json("NSM_H/CWQTestResult.json")
ours_nsm_results = read_json("NSM_H/OursCWQTestResult.json")
test_items = load_ds_items(
    "data/dataset/CWQ/ComplexWebQuestions_test.json",
    "data/dataset/CWQ/CWQ_full_with_int_id.jsonl",
)
out_f = "out.jsonl"

for item, raw, ours in tzip(test_items, raw_nsm_results, ours_nsm_results):
    temp = {"question": item.question, "paths": []}
    if raw["hit"] == 0 and ours["hit"] == 1:
        top1_candi = "ns:" + raw["candidate"][0][0]
        for topic in item.topic_ents:
            paths = []
            forward_search(
                topic,
                set([top1_candi]),
                [topic],
                paths,
                freebase,
                item.get_question_key_lexical(),
            )
            temp["paths"] += paths
    append_jsonl(temp, out_f)
