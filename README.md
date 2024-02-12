
# EPR-KGQA: Complex Questions Answering over Knowledge Graph via Evidence Pattern Retrieval

Project for the WWW'24 paper: *Enhancing Complex Question Answering over Knowledge Graphs through Evidence Pattern Retrieval*

[![image](https://img.shields.io/badge/Paper-preprint_on_arXiv-blue.svg)](https://arxiv.org/abs/2402.02175)
[![image](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://github.com/nju-websoft/EPR-KGQA/tree/master?tab=Apache-2.0-1-ov-file)

## Table of Contents
- [Overview](#overview)
  - [Evidence pattern retrieval (EPR)](#evidence-pattern-retrieval-epr)
  - [Experimental Results](#experimental-results)
- [Project Organization](#project-organization)
- [Reproducing the Results](#reproducing-the-results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Overview

- EPR-KGQA is
  - an *information retrieval* style KGQA system to explicitly model the structural dependency via **evidence pattern retrieval**.
  - the best-performing method under *the supervision of only question-answer pairs* on *ComplexWebquestions* (as of 2024-02).
    - a method that does not rely on manually annotated formal queries or relation paths.


| ![image](https://github.com/nju-websoft/EPR-KGQA/assets/10251079/872de89d-409a-42cd-b1f9-f18d7ce191d6) |
| --- |
| Facts about the question *“What country, containing Stahuis, does Germany border?”*. The correct answer `Netherlands` is <ins>underlined</ins>. The noisy answer `Austria` does not contain `Stahuis`, but the names of relations connecting them express similar meanings. Systems insensitive to structural dependencies may be confused by the noises. |

### Evidence pattern retrieval (EPR)

  - We implement EPR by indexing the *atomic adjacency pattern* (AP) of resource pairs.
    <img src="https://github.com/nju-websoft/EPR-KGQA/assets/10251079/2b02a3f6-ace1-421a-b2dc-6019f6da483d" alt="image" width="50%" height="auto">

  - We enumerate the combinations of retrieved APs to construct candidate evidence patterns (EP).
    <img src="https://github.com/nju-websoft/EPR-KGQA/assets/10251079/31838c50-d3e5-4a33-9003-de7f957eb32b" alt="image" width="50%" height="auto">

  - Candidate EPs are scored using the `BERT-base` model, and the best one is selected to extract a subgraph for answer reasoning.

### Experimental Results

<img src="https://github.com/nju-websoft/EPR-KGQA/assets/10251079/89e05edf-3dc7-449d-bad9-d7c34908e9a1" alt="image" width="50%" height="auto">

The best results of IR methods are in **bold**, and the second-best results are <ins>underlined</ins>. **†** denotes that the method requires gold query annotation of all training questions. **∗** denotes few-shot methods.


## Project Organization
  - config.py
  - config_CWQ.yaml
  - config_WebQSP.yaml
  - preprocess
    - adjacent_info_prepare.py
    - heuristic_path_search.py
    - do_preprocess.py
  - atomic_pattern_retrieval
    - generate_positive_rr_aps_by_cached_paths.py
    - generate_training_data_for_ap_retrieval.py
    - train_biencoder.sh
    - biencoder
      - biencoder.py
      - run_biencoder.py
      - faiss_indexer.py
      - biencoder_inference.py
  - evidence_pattern_retrieval
    - ep_size_threshold.py
    - ep_construction.py
    - generate_candidate_eps.py
    - generate_ep_ranking_data.py
    - train_ep_ranking.sh
    - predict_ep_ranking.sh
    - BERT_Ranker
      - model_config.py
      - BertRanker.py
      - train_bert_ranker.py
  - subgraph_extraction
    - subgraph_extraction.py
    - convert_to_nsm_input.py
  - my_utils
    - fact.py
    - freebase.py
    - data_item.py
    - io_utils.py
    - logger.py
    - rel_base.py
    - ap_utils.py
    - ep_utils.py
  - data
    - dataset
      - CWQ
        - ComplexWebQuestions_train.json
        - ComplexWebQuestions_dev.json
        - ComplexWebQuestions_test.json
        - CWQ_full_with_int_id.jsonl
      - WebQSP
        - train_simple.jsonl
        - dev_simple.jsonl
        - test_simple.jsonl
        - WebQSP.train.json
        - WebQSP.test.json
    - cache
      - relation_info_fb.json
      - type_info_fb.json
      - rel_conn_fb.jsonl
      - rr_aps_fb.json
      - rr_aps_forward_reverse_dict.json
      - rr_aps_tag_dict.json
      - CWQ
        - cached_paths.jsonl
      - WebQSP 
        - cached_paths.jsonl
    - CWQ
      - ap_retrieval
      - ep_retrieval
      - subgraph_extraction
    - WebQSP
      - ap_retrieval
      - ep_retrieval
      - subgraph_extraction
  - NSM_H


## Reproducing the Results

### Preprocessing

#### CWQ
```
cd EPR_KGQA
export PYTHONPATH=.
python preprocess/do_preprocess.py
```
#### WebQSP
```
cd EPR_KGQA
export PYTHONPATH=.
python preprocess/do_preprocess.py --config config_WebQSP.yaml
```

### Atomic Pattern Retrieval
#### CWQ
```
cd EPR_KGQA
export PYTHONPATH=.
python atomic_pattern_retrieval/generate_positive_rr_aps_by_cached_paths.py
python atomic_pattern_retrieval/generate_training_data_for_ap_retrieval.py
chmod +x atomic_pattern_retrieval/train_biencoder.sh
sh -x atomic_pattern_retrieval/train_biencoder.sh CWQ # for CWQ, about 5 hours per epoch, which costs about 1 day in total.
python atomic_pattern_retrieval/biencoder/biencoder_inference.py
```
#### WebQSP
```
cd EPR_KGQA
export PYTHONPATH=.
python atomic_pattern_retrieval/generate_positive_rr_aps_by_cached_paths.py --config config_WebQSP.yaml
python atomic_pattern_retrieval/generate_training_data_for_ap_retrieval.py --config config_WebQSP.yaml
chmod +x atomic_pattern_retrieval/train_biencoder.sh 
sh -x atomic_pattern_retrieval/train_biencoder.sh WebQSP
python atomic_pattern_retrieval/biencoder/biencoder_inference.py --config config_WebQSP.yaml
```
### Evidence Pattern Retrieval
#### CWQ
```
cd EPR_KGQA
export PYTHONPATH=.
python evidence_pattern_retrieval/generate_candidate_eps.py # ep construction
python evidence_pattern_retrieval/generate_ep_ranking_data.py
chmod +x evidence_pattern_retrieval/train_ep_ranking.sh
sh -x evidence_pattern_retrieval/train_ep_ranking.sh CWQ # about 2 days
chmod +x evidence_pattern_retrieval/predict_ep_ranking.sh
CUDA_VISIBLE_DEVICES=0 sh -x evidence_pattern_retrieval/predict_ep_ranking.sh CWQ test 7 100 # ds_tag, split, epoch, topk 
CUDA_VISIBLE_DEVICES=0 sh -x evidence_pattern_retrieval/predict_ep_ranking.sh CWQ dev 7 100 # ds_tag, split, epoch, topk 
CUDA_VISIBLE_DEVICES=0 sh -x evidence_pattern_retrieval/predict_ep_ranking.sh CWQ train 7 100 # ds_tag, split, epoch, topk
CUDA_VISIBLE_DEVICES=0 sh -x evidence_pattern_retrieval/predict_ep_ranking.sh CWQ test 7 80# ds_tag, split, epoch, topk 
```
#### WebQSP
```
cd EPR_KGQA
export PYTHONPATH=.
python evidence_pattern_retrieval/generate_candidate_eps.py --config config_WebQSP.yaml
python evidence_pattern_retrieval/generate_ep_ranking_data.py --config config_WebQSP.yaml
chmod +x evidence_pattern_retrieval/train_ep_ranking.sh 
sh -x evidence_pattern_retrieval/train_ep_ranking.sh WebQSP # about 2 hours
chmod +x evidence_pattern_retrieval/predict_ep_ranking.sh
CUDA_VISIBLE_DEVICES=0 sh -x evidence_pattern_retrieval/predict_ep_ranking.sh WebQSP test 6 100 # ds_tag, split, epoch, topk 
CUDA_VISIBLE_DEVICES=0 sh -x evidence_pattern_retrieval/predict_ep_ranking.sh WebQSP dev 6 100 # ds_tag, split, epoch, topk 
CUDA_VISIBLE_DEVICES=0 sh -x evidence_pattern_retrieval/predict_ep_ranking.sh WebQSP train 6 100 # ds_tag, split, epoch, topk
CUDA_VISIBLE_DEVICES=0 sh -x evidence_pattern_retrieval/predict_ep_ranking.sh WebQSP test 6 80 # ds_tag, split, epoch, topk 
```
### Subgraph Extraction
#### CWQ
```
cd EPR_KGQA
export PYTHONPATH=.
python subgraph_extraction/subgraph_extraction.py
python subgraph_extraction/convert_to_nsm_input.py
```
#### WebQSP
```
cd EPR_KGQA
export PYTHONPATH=.
python subgraph_extraction/subgraph_extraction.py --config config_WebQSP.yaml
python subgraph_extraction/convert_to_nsm_input.py --config config_WebQSP.yaml
```

### NSM Reasoning
#### CWQ

```
cd NSM_H
export PYTHONPATH=.
chmod +x ../answer_reasoning/train_nsm.sh
sh -x ../answer_reasoning/train_nsm.sh CWQ
chmod +x ../answer_reasoning/predict_nsm.sh
sh -x ../answer_reasoning/predict_nsm.sh CWQ
```
#### WebQSP

```
cd NSM_H
export PYTHONPATH=.
chmod +x ../answer_reasoning/train_nsm.sh
sh -x ../answer_reasoning/train_nsm.sh WebQSP
chmod +x ../answer_reasoning/predict_nsm.sh
sh -x ../answer_reasoning/predict_nsm.sh WebQSP
```

## Citation
```
@inproceedings{epr-kgqa,
  author = {Ding, Wentao and Li, Jinmao and Luo, Liangchuan and Qu, Yuzhong},
  title = {Enhancing Complex Question Answering over Knowledge Graphs through Evidence Pattern Retrieval},
  year = {2024},
  booktitle = {Proceedings of the ACM Web Conference 2024},
  series = {WWW '24}
}
```

## Acknowledgements
Our project uses [WSDM2021_NSM (the Neural State Machine for KBQA)](https://github.com/RichardHGL/WSDM2021_NSM) as the answer reasoner.
