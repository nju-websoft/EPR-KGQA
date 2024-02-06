# EPR for WWW2024
Research Project for Evidence Pattern Retrieval for the Web Conference 2024
[paper (preprint on arXiv)](https://arxiv.org/abs/2402.02175)

## 项目结构：
  - my_utils
    - freebase.py
    - data_item.py
    - io_utils.py
    - logger.py
    - rel_base.py
    - ap_utils.py
    - ep_utils.py
  - preprocess
    - heuristic_path_search.py
    - adjacent_info_prepare.py
  - evidence_pattern_retrieval
    - models
      - ???
      - ???
    - ap_retrieval_td_gen.py
    - ep_rank_td_gen.py
    - ep_construction.py
  - data
    - APs_fb
    - WebQSP
      - dataset
      - ap_retrieval
      - ep_construction
      - ep_ranking
      - NSM_ours
    - CWQ
      - dataset
      - ap_retrieval
      - ep_construction
      - ep_ranking
      - NSM_ours
  - NSM
  - config.py
  - config_CWQ.yaml
  - config_WebQSP.yaml
  - do_preprocess.py
  - do_training.py
  - do_inference.py
  - run_nsm.ipynb
  - analyze_result.ipynb
