import sys

import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer



def l2_normalize(tensor, axis=-1):
    """L2-normalize columns of tensor"""
    return F.normalize(tensor, p=2, dim=axis)


class EncoderBert(nn.Module):
    def __init__(self, embed_dim, txt_type):
        super(EncoderBert, self).__init__()
        if txt_type == 'bert-base-uncased':
            self.txt_enc = BertModel.from_pretrained("bert-base-uncased")
            # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.linear = nn.Linear(768, embed_dim)
        elif txt_type == 'tinyBert':
            self.txt_enc = BertModel.from_pretrained("huawei-noah/TinyBERT_4L_zh")
            # self.tokenizer = BertTokenizer.from_pretrained("huawei-noah/TinyBERT_4L_zh")
            self.linear = nn.Linear(312, embed_dim) 

    def forward(self, tokenizer,sentences):
        inputs = tokenizer(sentences, padding="max_length", return_tensors='pt', truncation=True, max_length=128)
        for a in inputs:
            inputs[a] = inputs[a].cuda()
        out = self.txt_enc(**inputs)
        out = {'embedding': l2_normalize(self.linear(out['last_hidden_state'][:, 0, :]))}  # [bsz, 768]
        return out