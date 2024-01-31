# Stardard Libraries
import os
import time

# Third party libraries
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tqdm.contrib import tzip
from transformers import AutoTokenizer

# Self-defined Modules
from biencoder import BiEncoderModule
from config import Config
from faiss_indexer import DenseFlatIndexer
from my_utils.data_item import load_ds_items
from my_utils.freebase import FreebaseODBC
from my_utils.io_utils import read_json, write_json
from my_utils.logger import Logger

logger = Logger.get_logger("biencoder_inference")


class CustomDataset(Dataset):
    def __init__(
        self, rr_aps, maxlen, tokenizer=None, bert_model="bert-base-uncased"
    ):
        self.rr_aps = rr_aps
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(bert_model)
        )
        self.maxlen = maxlen

    def __len__(self):
        return len(self.rr_aps)

    def __getitem__(self, index):
        rr_ap = self.rr_aps[index]
        encoded_rr_ap = self.tokenizer(
            rr_ap,
            padding="max_length",
            truncation=True,
            max_length=self.maxlen,
            return_tensors="pt",
        )
        rr_ap_token_ids = encoded_rr_ap["input_ids"].squeeze(
            0
        )  # tensor of token ids
        rr_ap_attn_masks = encoded_rr_ap["attention_mask"].squeeze(
            0
        )  # binary tensor with "0" for padded values and "1" for the other values
        rr_ap_token_type_ids = encoded_rr_ap["token_type_ids"].squeeze(
            0
        )  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        return rr_ap_token_ids, rr_ap_attn_masks, rr_ap_token_type_ids


def encode_rr_aps(
    rr_aps,
    model_path,
    inference_dir,
    save_path,
    max_len=32,
    batch_size=128,
    cache_dir="bert-base-uncased",
):
    if os.path.exists(os.path.join(inference_dir, "flat.index")):
        print("flat index already exists!")
        return
    if os.path.exists(save_path):
        print(f"{save_path} already exists!")
        return
    maxlen = max_len
    bs = batch_size
    bert_model = cache_dir

    tokenizer = AutoTokenizer.from_pretrained(bert_model)

    # device="cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiEncoderModule(
        device,
        bert_model=bert_model,
        tokenizer=tokenizer if tokenizer else None,
        freeze_bert=True,
    )

    print("Loading the weights of the model...")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    rr_aps_set = CustomDataset(
        rr_aps,
        maxlen,
        bert_model=bert_model,
        tokenizer=tokenizer if tokenizer else None,
    )
    rr_aps_loader = DataLoader(rr_aps_set, batch_size=bs, num_workers=1)

    rr_ap_vectors = torch.zeros(0).to(device)
    with torch.no_grad():
        for rr_ap_token_ids, rr_ap_attn_masks, rr_ap_token_type_ids in tqdm(
            rr_aps_loader
        ):
            # print('rr_ap_token_ids: {}'.format(rr_ap_token_ids.shape))
            embedded_rr_ap = model.encode_relation(
                rr_ap_token_ids.to(device),
                rr_ap_attn_masks.to(device),
                rr_ap_token_type_ids.to(device),
            )
            rr_ap_vectors = torch.cat((rr_ap_vectors, embedded_rr_ap), 0)
    print("rr_ap_vectors: {}".format(rr_ap_vectors.shape))
    torch.save(rr_ap_vectors, save_path)


def build_index(output_path, rr_ap_vectors_path, index_buffer=50000):
    """
    index_buffer: Temporal memory data buffer size (in samples) for indexer
    """
    if os.path.exists(output_path):
        print(f"{output_path} already exists!")
        return
    output_dir, _ = os.path.split(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Loading rr_ap vectors from path: %s" % rr_ap_vectors_path)
    rr_ap_vectors = torch.load(rr_ap_vectors_path).cpu().detach().numpy()
    vector_size = rr_ap_vectors.shape[1]

    logger.info("Using Flat index in FAISS")
    index = DenseFlatIndexer(vector_size, index_buffer)
    # logger.info("Using HNSW index in FAISS")
    # index = DenseHNSWFlatIndexer(vector_size, index_buffer)

    logger.info("Building index.")
    index.index_data(rr_ap_vectors)
    logger.info("Done indexing data.")

    index.serialize(output_path)


def inference_pipeline(
    questions_path,
    all_rr_aps,
    model_path,
    max_len,
    cache_dir,
    dest_file,
    index_file,
    vector_size=768,
    index_buffer=50000,
    top_k=500,
):
    if os.path.exists(dest_file):
        print(f"{dest_file} already exists!")
        return
    maxlen = max_len
    bs = 256
    bert_model = cache_dir
    print(questions_path)

    tokenizer = AutoTokenizer.from_pretrained(bert_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiEncoderModule(
        device, bert_model=bert_model, tokenizer=tokenizer, freeze_bert=True
    )

    print("Loading the weights of the model...")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    data = load_ds_items(questions_path)
    questions = [data_item.question.lower() for data_item in data]
    questions_set = CustomDataset(
        questions, maxlen, bert_model=bert_model, tokenizer=tokenizer
    )
    questions_loader = DataLoader(questions_set, batch_size=bs, num_workers=2)

    questions_vectors = torch.zeros(0).to(device)
    with torch.no_grad():
        for question_token_ids, question_attn_masks, question_token_type_ids in tqdm(
            questions_loader
        ):
            # print('rr_ap_token_ids: {}'.format(rr_ap_token_ids.shape))
            embedded_question = model.encode_question(
                question_token_ids.to(device),
                question_attn_masks.to(device),
                question_token_type_ids.to(device),
            )
            questions_vectors = torch.cat((questions_vectors, embedded_question), 0)
    print("question_vectors: {}".format(questions_vectors.shape))
    index = DenseFlatIndexer(vector_size, index_buffer)
    index.deserialize_from(index_file)
    question_vectors = questions_vectors.cpu().detach().numpy()
    _, pred_rr_ap_indexes = index.search_knn(question_vectors, top_k=top_k)
    pred_rr_aps = [
        list([all_rr_aps[index] for index in indexes])
        for indexes in pred_rr_ap_indexes
    ]
    dest_data = []
    for rr_aps, data_item in tzip(pred_rr_aps, data):
        topic_ents = data_item.topic_ents
        er_aps = []
        for te in topic_ents:
            neighbor_rels = FreebaseODBC.query_neighbor_rels([te])
            for rel in neighbor_rels:
                if rel.endswith("_Rev"):
                    er_aps.append(" ".join([te, "rev", rel[:-4]]))
                else:
                    er_aps.append(" ".join([te, "fwd", rel]))
        dest_item = {
            "id": data_item.id,
            "question": data_item.question,
            "rr_aps": rr_aps,
            "er_aps": er_aps,
        }
        dest_data.append(dest_item)
    write_json(dest_data, dest_file)


if __name__ == "__main__":
    topk = 500
    epoch = 5
    inference_dir = os.path.join(
        f"data/{Config.ds_tag}/ap_retrieval/model", f"{Config.ds_tag}_ep_{epoch}"
    )
    model_file = os.path.join(
        f"data/{Config.ds_tag}/ap_retrieval/model", f"{Config.ds_tag}_ep_{epoch}.pt"
    )
    epoch_folder = inference_dir.split("/")[-1]
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)
    encode_rr_aps(
        rr_aps=read_json(Config.cache_rr_aps),
        model_path=model_file,
        inference_dir=inference_dir,
        save_path=os.path.join(inference_dir, "rr_aps_fb.pt"),
        max_len=95,  # consistent with bi-encoder training script
        batch_size=128,
        cache_dir="bert-base-uncased",
    )
    build_index(
        output_path=os.path.join(inference_dir, "flat.index"),
        rr_ap_vectors_path=os.path.join(inference_dir, "rr_aps_fb.pt"),
    )
    for split in ["dev", "test", "train"]:
        start = time.time()
        inference_pipeline(
            questions_path=Config.ds_split_f(split),
            all_rr_aps=read_json(Config.cache_rr_aps),
            model_path=model_file,
            max_len=95,  # consistent with bi-encoder training script
            cache_dir="bert-base-uncased",
            dest_file=os.path.join(
                Config.retrieved_ap_f(split),
            ),
            index_file=os.path.join(inference_dir, "flat.index"),
            top_k=topk
        )
        end = time.time()
        print(end - start)
