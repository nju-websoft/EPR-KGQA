"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import BertPreTrainedModel, BertModel


def get_inf_mask(bool_mask):
    return (~bool_mask) * -100000.0


class BertForCandidateRanking(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    # for training return loss, [batch_size * num_sample]
    # for testing, batch size have to be 1
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sample_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        assert return_dict is None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # for training, input is batch_size * sample_size * L
        # for testing, it is batch_size * L
        if labels is not None:
            batch_size = input_ids.size(0)
            sample_size = input_ids.size(1)
            seq_length = input_ids.size(2)
            # flatten first two dim
            input_ids = input_ids.view((batch_size * sample_size, -1))
            token_type_ids = token_type_ids.view((batch_size * sample_size, -1))
            attention_mask = attention_mask.view((batch_size * sample_size, -1))

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # embedding_by_tokens = outputs[0]
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # embedding_by_tokens = embedding_by_tokens.view((batch_size, sample_size, seq_length, 768))

        loss = None
        if labels[0].item() != -1:
            # reshape logits
            logits = logits.view((batch_size, sample_size))
            logits = logits + get_inf_mask(sample_mask)
            # apply infmask
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
        else:
            logits = logits.view((batch_size, sample_size))
            logits = logits + get_inf_mask(sample_mask)

        return (loss, logits) if loss is not None else logits


class ListDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

    def __iter__(self):
        return iter(self.examples)


# for single problem
class RankingFeature:
    def __init__(self, pid, input_ids, token_type_ids, target_idx):
        self.pid = pid
        self.candidate_input_ids = input_ids
        self.candidate_token_type_ids = token_type_ids
        self.target_idx = target_idx


def _collect_contrastive_inputs(feat, num_sample, dummy_inputs):
    input_ids = []
    token_type_ids = []

    input_ids.extend(feat.candidate_input_ids)
    token_type_ids.extend(feat.candidate_token_type_ids)
    filled_num = len(input_ids)
    # force padding
    for _ in range(filled_num, num_sample):
        input_ids.append(dummy_inputs['input_ids'])
        token_type_ids.append(dummy_inputs['token_type_ids'])
    sample_mask = [1] * filled_num + [0] * (num_sample - filled_num)
    return input_ids, token_type_ids, sample_mask


def disamb_collate_fn(data, tokenizer):
    dummy_inputs = tokenizer('', '', return_token_type_ids=True)
    # batch size
    # input_id: B * N_Sample * L
    # token_type: B * N_Sample * L
    # attention_mask: B * N_Sample * N
    # sample_mask: B * N_Sample
    # labels: B, all zero
    batch_size = len(data)
    num_sample = max([len(x.candidate_input_ids) for x in data])

    all_input_ids = []
    all_token_type_ids = []
    all_sample_masks = []
    for feat in data:
        input_ids, token_type_ids, sample_mask = _collect_contrastive_inputs(feat, num_sample, dummy_inputs)
        all_input_ids.extend(input_ids)
        all_token_type_ids.extend(token_type_ids)
        all_sample_masks.append(sample_mask)

    encoded = tokenizer.pad({'input_ids': all_input_ids, 'token_type_ids': all_token_type_ids}, return_tensors='pt')
    all_sample_masks = torch.BoolTensor(all_sample_masks)
    labels = torch.LongTensor([x.target_idx for x in data])

    all_input_ids = encoded['input_ids'].view((batch_size, num_sample, -1))
    all_token_type_ids = encoded['token_type_ids'].view((batch_size, num_sample, -1))
    all_attention_masks = encoded['attention_mask'].view((batch_size, num_sample, -1))
    return all_input_ids, all_token_type_ids, all_attention_masks, all_sample_masks, labels
