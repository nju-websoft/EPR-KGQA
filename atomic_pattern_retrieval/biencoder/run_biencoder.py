# Stardard Libraries
import argparse
import copy
import os
import random

# Third party libraries
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score

# Self-defined Modules
from biencoder import BiEncoderModule


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', default="CWQ", type=str, help="CWQ | WebQSP")
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--dev_file', type=str, required=True)
    parser.add_argument('--model_save_path', default='data/', type=str)
    parser.add_argument('--batch_size', default=4, type=int, help="4 for CWQ")
    parser.add_argument('--epochs', default=1, type=int, help="1 for CWQ, 3 for WebQSP")
    parser.add_argument('--log_dir', default='log/', type=str)
    parser.add_argument('--cache_dir', default='bert-base-uncased', type=str)
    parser.add_argument('--add_special_tokens', default=False, action='store_true',
                        help='whether to add special tokens')
    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def evaluate(model, device, dataloader):
    model.eval()

    mean_loss = 0
    count = 0
    golden_truth = []
    preds = []

    with torch.no_grad():
        for question_token_ids, question_attn_masks, question_token_type_ids, relations_token_ids, relations_attn_masks, \
            relations_token_type_ids, golden_id in tqdm(dataloader):
            scores, loss = model(
                question_token_ids.to(device),
                question_attn_masks.to(device),
                question_token_type_ids.to(device),
                relations_token_ids.to(device),
                relations_attn_masks.to(device),
                relations_token_type_ids.to(device),
                golden_id.to(device)
            )
            mean_loss += loss
            count += 1
            pred_id = torch.argmax(scores, dim=1)
            # print('pred_id: {}'.format(pred_id.shape))
            # print('golden_id: {}'.format(golden_id.shape))
            preds += pred_id.tolist()
            golden_truth += golden_id.tolist()

    accuracy = accuracy_score(golden_truth, preds)

    return mean_loss / count, accuracy


class CustomDataset(Dataset):
    def __init__(self, data_file):
        self.data = torch.load(data_file)

    def __len__(self):
        return int(len(self.data))

    def __getitem__(self, index):
        result = self.data[index]
        return result['question_tokens_ids'], result['question_attn_masks'], result['question_token_type_ids'], result[
            'relation_tokens_ids'], result['relation_attn_masks'], result['relation_token_type_ids'], result[
                   'golden_id']


def train_bert(model, opti, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate, device, log_path,
               model_save_path, dataset_type):
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 5
    if log_path:
        log_w = open(log_path, 'w')
    scaler = GradScaler()
    best_loss = np.Inf
    best_epoch = 1

    for ep in range(epochs):
        model.train()
        running_loss = 0.0

        for it, (
                question_token_ids, question_attn_masks, question_token_type_ids, relations_token_ids,
                relations_attn_masks,
                relations_token_type_ids, golden_id) in enumerate(tqdm(train_loader)):
            scores, loss = model(
                question_token_ids.to(device),
                question_attn_masks.to(device),
                question_token_type_ids.to(device),
                relations_token_ids.to(device),
                relations_attn_masks.to(device),
                relations_token_type_ids.to(device),
                golden_id.to(device)
            )
            loss = loss / iters_to_accumulate
            scaler.scale(loss).backward()

            if (it + 1) % iters_to_accumulate == 0:
                scaler.step(opti)
                # Updates the scale for next iteration.
                scaler.update()
                # Adjust the learning rate based on the number of iterations.
                lr_scheduler.step()
                # Clear gradients
                opti.zero_grad()

            running_loss += loss.item()
            if (it + 1) % print_every == 0:  # Print training loss information
                print()
                print("Iteration {}/{} of epoch {} complete. Loss : {} "
                      .format(it + 1, nb_iterations, ep + 1, running_loss / print_every))

                running_loss = 0.0

        if val_loader:
            val_loss, accuracy = evaluate(model, device, val_loader)
            print("Epoch {} complete! Validation Loss : {}".format(ep + 1, val_loss))
            print("Accuracy on dev data: {}\n".format(accuracy))
            if log_w:
                log_w.write("Epoch {} complete! Validation Loss : {}\n".format(ep + 1, val_loss))
                log_w.write("Accuracy on dev data: {}\n".format(accuracy))
        # Recording validation loss, while still saving models of every epoch
        model_copy = copy.deepcopy(model)
        if val_loss < best_loss:
            print("Best validation loss improved from {} to {}".format(best_loss, val_loss))
            print()
            best_loss = val_loss
            best_epoch = ep + 1

        model_path = os.path.join(model_save_path, '{}_ep_{}.pt'.format(dataset_type, ep + 1))
        torch.save(model_copy.state_dict(), model_path)
        print("The model has been saved in {}".format(model_path))

    if log_w:
        log_w.close()
    print('Best epoch is: {}, with validation loss: {}'.format(best_epoch, best_loss))
    del loss
    torch.cuda.empty_cache()


def main(args):
    bert_model = args.cache_dir
    freeze_bert = False
    bs = args.batch_size
    iters_to_accumulate = 2  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
    lr = 2e-5  # learning rate
    epochs = args.epochs
    log_path = os.path.join(args.log_dir, 'log.txt')
    set_seed(1)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiEncoderModule(device, bert_model='bert-base-uncased', tokenizer=tokenizer, freeze_bert=freeze_bert)

    if args.add_special_tokens:
        special_tokens_dict = {'additional_special_tokens': ['S-S', 'O-O', 'S-O']}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.question_bert_layer.resize_token_embeddings(len(tokenizer))
        model.relation_bert_layer.resize_token_embeddings(len(tokenizer))

    if args.cache_dir != "bert-base-uncased":
        model.load_state_dict(torch.load(bert_model))
    model.to(device)

    opti = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    num_warmup_steps = 0  # The number of steps for the warmup phase.

    print("Reading training data...")
    train_set = CustomDataset(args.train_file)
    train_loader = DataLoader(train_set, batch_size=bs, num_workers=2)
    if args.dev_file is not None:
        print("Reading validation data...")
        val_set = CustomDataset(args.dev_file)
        val_loader = DataLoader(val_set, batch_size=bs, num_workers=2)
    else:
        val_loader = None

    t_total = (
                      len(train_loader) // iters_to_accumulate) * epochs  # Necessary to take into account Gradient accumulation
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps,
                                                   num_training_steps=t_total)

    train_bert(model, opti, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate, device, log_path,
               args.model_save_path, args.dataset_type)


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    main(args)
