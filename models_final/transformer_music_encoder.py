# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" Define the Seq2Seq Generation Network """
import numpy as np
import torch
import torch.nn as nn
# from v2.utils.pose import BOS_POSE
from models_final.layers import MultiHeadAttention, PositionwiseFeedForward


def get_non_pad_mask(seq):
    assert seq.dim() == 3
    non_pad_mask = torch.abs(seq).sum(2).ne(0).type(torch.float)
    return non_pad_mask.unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """
    len_q = seq_q.size(1)
    padding_mask = torch.abs(seq_k).sum(2).eq(0)  # sum the vector of last dim and then judge
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq, sliding_windown_size):
    """ For masking out the subsequent info. """
    batch_size, seq_len, _ = seq.size()
    mask = torch.ones((seq_len, seq_len), device=seq.device, dtype=torch.uint8)
    mask = torch.triu(mask, diagonal=-sliding_windown_size)
    mask = torch.tril(mask, diagonal=sliding_windown_size)
    mask = 1 - mask
    # print(mask)
    return mask.bool()


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, d_condition=None):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout, d_condition=d_condition)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout, d_condition=d_condition)

    def forward(self, enc_input, slf_attn_mask=None, non_pad_mask=None, condition=None):
        if not isinstance(condition, list):
            condition_attn = condition
            condition_ffn = condition
        else:
            condition_attn = condition[0]
            condition_ffn = condition[1]
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask, condition=condition_attn)
        # enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output, condition=condition_ffn)
        # enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class MusicEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, max_seq_len=100, input_size=32, d_word_vec=200,
            n_layers=3, n_head=8, d_k=64, d_v=64,
            d_model=200, d_inner=1024, dropout=0.1):
            # self, max_seq_len = 100, input_size = 35, d_word_vec = 640,
            # n_layers = 3, n_head = 10, d_k = 64, d_v = 64,
            # d_model = 640, d_inner = 1920, dropout = 0.1):

        super().__init__()

        self.d_model = d_model
        n_position = max_seq_len + 1

        self.src_emb = nn.Linear(input_size, d_word_vec)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.outlen = d_model

    def get_outlen(self):
        return self.outlen

    def forward(self, src_seq, mask=None, return_attns=False):

        enc_slf_attn_list = []
        src_pos = torch.tensor([i for i in range(src_seq.shape[1])]).unsqueeze(0).cuda()

        # -- Forward
        enc_output = self.src_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=mask)
    
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output
