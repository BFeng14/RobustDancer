# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" Define the attention layers. """
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            batch_size, _, _ = attn.size()
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, d_condition=None):
        super().__init__()

        assert d_k == d_v
        # assert d_v * n_head == d_model

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.d_condition = d_condition
        if d_condition is not None:
            self.scale = ScaleOffset(d_model, d_condition, d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, condition=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output)
        if condition is not None and self.d_condition is not None:
            output = self.scale(output, condition)

        return output + residual, attn


class ScaleOffset(nn.Module):

    """Conditional Batch Normalization"""

    def __init__(self, num_features, condition_size, hidden_size):
        super(ScaleOffset, self).__init__()
        # beta and gamma parameters for each channel - defined as trainable parameters
        self.betas = nn.Parameter(torch.zeros(num_features))
        self.gammas = nn.Parameter(torch.ones(num_features))
        if hidden_size is None:
            hidden_size = num_features  # * 4
        # MLP used to predict betas and gammas
        self.fc_gamma = nn.Sequential(
            nn.Linear(condition_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_features),
        )

        self.fc_beta = nn.Sequential(
            nn.Linear(condition_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_features),
        )

        # initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, input, conditions):
        """
        :param input: [b, len, f]
        :param kwargs:
        :return:
        """
        gammas = self.fc_gamma(conditions)  # [b, f]
        betas = self.fc_beta(conditions)  # [b, f]
        if gammas.dim() == 1:
            gammas = gammas.unsqueeze(0)
        if betas.dim() == 1:
            betas = betas.unsqueeze(0)
        size = input.size()
        gammas += self.gammas
        betas += self.betas
        gammas = gammas.unsqueeze(1).expand(size)
        betas = betas.unsqueeze(1).expand(size)
        return gammas * input + betas


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1, d_condition=None):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)

        self.d_condition = d_condition
        if d_condition is not None:
            self.scale = ScaleOffset(d_in, d_condition, d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, condition=None):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output)
        if condition is not None and self.d_condition is not None:
            output = self.scale(output, condition)
        return output + residual
