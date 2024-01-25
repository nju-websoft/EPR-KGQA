# Third party libraries
from tqdm import tqdm

# Self-defined Modules
from config import Config
from my_utils.ep_utils import get_instantiable_topk, integrate_topk_instantiated_subg
from my_utils.io_utils import read_json, write_json

Config.ap_topk = 100
test_ranked_eps = read_json(Config.ranked_ep_f("test"))
test_ranked_ans = read_json(f"NSM_H/Ours{Config.ds_tag}TestResult.json")

error_info = []
for epinfo, ansinfo in tqdm(zip(test_ranked_eps, test_ranked_ans)):
    qid = epinfo["ID"]
    qtext = epinfo["question"]
    gAns = set(["ns:" + item for item in ansinfo["goldAns"]])
    if ansinfo["hit"] == 0:
        top10_hit = False
        top1_hit = False
        top10_eps = get_instantiable_topk(
            [item[0] for item in epinfo["sorted_candidates_with_logits"]], 10
        )
        top1_ep = top10_eps[:1]
        _, top10_node_eps, _ = integrate_topk_instantiated_subg(top10_eps, 10)
        _, top1_node_eps, _ = integrate_topk_instantiated_subg(top1_ep, 1)
        if len(gAns & set(top10_node_eps.keys())) > 0:
            top10_hit = True
        if len(gAns & set(top1_node_eps.keys())) > 0:
            top1_hit = True
        info = {
            "id": qid,
            "qtext": qtext,
            "top1_cover": top1_hit,
            "top10_cover": top10_hit,
            "top10_eps": top10_eps,
        }
        error_info.append(info)
write_json(error_info, f"{Config.ds_tag}_err_analysis.json")
