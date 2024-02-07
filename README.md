
# EPR-KGQA: Complex Questions Answering over Knowledge Graph via Evidence Pattern Retrieval

Project for the WWW'24 paper: *Enhancing Complex Question Answering over Knowledge Graphs through Evidence Pattern Retrieval*

[![image](https://img.shields.io/badge/Paper-preprint_on_arXiv-blue.svg)](https://arxiv.org/abs/2402.02175)
[![image](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://github.com/nju-websoft/EPR-KGQA/tree/master?tab=Apache-2.0-1-ov-file)

## Table of Contents
- [Overview](#overview)
  - [Evidence pattern retrieval (EPR)](#evidence-pattern-retrieval-epr)
  - [Experimental Results](#experimental-results)
- TODO
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
  //TODO
  
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
