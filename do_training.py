# 训练数据生成和训练过程启动都放在这个文件中

from evidence_pattern_retrieval.ep_rank_td_gen import run as ep_rank_td_gen, Config
from evidence_pattern_retrieval.ans_ranker_td_gen import run as ans_rank_td_gen

if __name__ == "__main__":
  # 训练数据生成阶段，统一设置 ap_topK=100, ep_topk=3
  raw_ap_topk = Config.ap_topk
  raw_ep_topk = Config.ep_topk
  Config.ap_topk = 100
  Config.ep_topk = 3
  
  # >>> atomic pattern retrieval td gen
  # 【TODO】
  
  # >>> atomic pattern retrieval training
  # 【TODO】
  
  # >>> evidence pattern rank td gen
  # ep_rank_td_gen()
  
  # >>> evidence pattern rank training
  # 【TODO】
  
  # >>> generate induced subgraphs for answer ranking module
  ans_rank_td_gen()
  
  # 训练阶段结束后，还原 Config 变量值
  # Config.ap_topk = raw_ap_topk
  # Config.ep_topk = raw_ep_topk