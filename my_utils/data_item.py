# Stardard Libraries
from typing import List, Set

# Third party libraries
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Self-defined Modules
from config import Config
from my_utils.io_utils import *


class DataItem:
    # nltk
    stop_words = set(stopwords.words("english")) | set([".", ",", "?"])
    lemmatizer = WordNetLemmatizer()

    def __init__(
        self,
        question_id: str,
        question: str,
        topic_ents: List[str],
        answers: List[str],
        lf=None,
        comp_type=None,
    ):
        self.id = question_id
        self.question = question
        self.topic_ents = topic_ents
        self.answers = answers
        self.lf = lf
        self.comp_type = comp_type
        self.dataset = Config.ds_tag

    def get_question_key_lexical(self) -> Set[str]:
        tokens = set()
        for token in word_tokenize(self.question):
            token = token.lower()
            if token not in self.stop_words:
                tokens.add(token)
        lemmas = self.get_lemmas(tokens)
        return tokens | lemmas

    @staticmethod
    def get_lemmas(tokens: Set[str]) -> Set[str]:
        return set([DataItem.lemmatizer.lemmatize(token) for token in tokens])


# 从数据集中批量初始化数据项
def load_ds_items(
    filepath: str, sr_info_file: str = Config.ds_sr_all
) -> List[DataItem]:
    results = []
    if Config.ds_tag == "CWQ":
        items = json.load(open(filepath))
        sr_info = read_jsonl_by_key(sr_info_file)
        for item in items:
            id = item["ID"]
            question = item["question"]
            topic_ents = []
            for info in sr_info[id]["entities"]:
                if info["kb_id"].endswith("?x"):
                    topic_ents.append("ns:" + info["kb_id"][:-2])
                else:
                    topic_ents.append("ns:" + info["kb_id"])
            answers = [
                "ns:" + answer["answer_id"]
                if answer["answer_id"].startswith("m.")
                or answer["answer_id"].startswith("g.")
                else answer["answer_id"]
                for answer in item["answers"]
            ]
            lf = item["sparql"]
            comp_type = item["compositionality_type"]
            results.append(DataItem(id, question, topic_ents, answers, lf, comp_type))
    elif Config.ds_tag == "WebQSP":
        items = read_jsonl_by_key(filepath)
        for id in items:
            item = items[id]
            question = item["question"]
            topic_ents = ["ns:" + mid for mid in item["entities"]]
            answers = list(set(["ns:" + info["kb_id"] for info in item["answers"]]))
            lf = None
            comp_type = None
            results.append(DataItem(id, question, topic_ents, answers, lf, comp_type))
    else:
        print("[WARN] supported datasets: CWQ | WebQSP ")
    return results
