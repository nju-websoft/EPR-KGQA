from config import Config
from my_utils.pred_base import PredBase
from preprocess.adjacent_info_prepare import AdjacentInfoPrepare
from preprocess.heuristic_path_search import run as heuristic_path_search

AdjacentInfoPrepare.load_global_cache()
heuristic_path_search()
