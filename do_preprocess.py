# Self-defined Modules
from preprocess.adjacent_info_prepare import AdjacentInfoPrepare
from preprocess.heuristic_path_search import run as heuristic_path_search

AdjacentInfoPrepare.load_global_cache()
heuristic_path_search()
