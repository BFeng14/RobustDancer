#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
# from LTD_utils import data_utils
from models_final.mag_model import MAG
import numpy as np
from utils import str2bool
from models_final.transformer_music_encoder import MusicEncoder, EncoderLayer, get_sinusoid_encoding_table


class DanceModel(nn.Module):
    def __init__(self, args):
        super(DanceModel, self).__init__()
        self.args = args
        self.use_music = args.use_music
        self.use_style = args.use_style
        self.use_hist = args.use_hist
        hidden_feature = args.hidden_feature
        p_dropout = args.p_dropout
        num_stage = args.num_stage
        node_n = args.node_n
        d_model = args.d_model
        n_position = 100

        self.num_stage = num_stage

        if self.use_music:
            self.music_encoder = MusicEncoder()
            self.music_linear = nn.Linear(200, d_model)

        self.src_emb = nn.Linear(node_n, d_model)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)
        self.beat_position_enc = nn.Embedding(n_position + 1, d_model)
        self.padding_embedding = nn.Embedding(4, d_model)

        self.beat_stack = nn.ModuleList([
            EncoderLayer(d_model, hidden_feature, args.n_head,
                         args.d_k, args.d_v,
                         dropout=p_dropout, d_condition=None)
            for _ in range(3)])

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, hidden_feature, args.n_head,
                         args.d_k, args.d_v,
                         dropout=p_dropout, d_condition=d_model)
            for _ in range(num_stage)])
        if self.use_style:
            self.style_stack = nn.ModuleList([
                EncoderLayer(d_model, hidden_feature, args.n_head,
                             args.d_k, args.d_v,
                             dropout=p_dropout, d_condition=None)
                for _ in range(3)])

        if self.use_hist:
            self.convQ = nn.Sequential(nn.Conv1d(in_channels=node_n, out_channels=d_model, kernel_size=6,
                                                 bias=False),
                                       nn.ReLU(),
                                       nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                                 bias=False),
                                       nn.ReLU())

            self.convK = nn.Sequential(nn.Conv1d(in_channels=node_n, out_channels=d_model, kernel_size=6,
                                                 bias=False),
                                       nn.ReLU(),
                                       nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                                 bias=False),
                                       nn.ReLU())
            self.softmax = nn.Softmax(dim=2)
            self.mag = MAG(d_model, d_model, d_model, d_model, beta_shift=0.1)


        self.pose_decode = nn.Linear(d_model, node_n)

        nn.init.normal_(self.padding_embedding.weight, mean=0, std=args.initializer_range)
        nn.init.normal_(self.beat_position_enc.weight, mean=0, std=args.initializer_range)

        self.l2loss = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        self.triplet_loss = nn.TripletMarginLoss(margin=args.triplet_margin)

    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument("--d_model", type=int, default=640)
        parser.add_argument("--d_k", type=int, default=64)
        parser.add_argument("--d_v", type=int, default=64)
        parser.add_argument("--n_head", type=int, default=10)
        parser.add_argument("--hidden_feature", type=int, default=1920)  # 256
        parser.add_argument("--p_dropout", type=float, default=0.1)
        parser.add_argument("--num_stage", type=int, default=12)
        parser.add_argument("--node_n", type=int, default=221)  # 18*3, 67
        parser.add_argument("--use_music", type=str2bool, default=True)
        parser.add_argument("--use_style", type=str2bool, default=True)
        parser.add_argument("--use_hist", type=str2bool, default=True)
        parser.add_argument("--fusion_type", type=str, default="cln")
        parser.add_argument("--condition_step", type=int, default=10)
        parser.add_argument("--lambda_v", type=float, default=0.03)
        parser.add_argument("--triplet_margin", type=float, default=3.)
        parser.add_argument("--initializer_range", type=float, default=0.02)
        return parser

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def run(self, x, music, hist, style, beat):
        ####################condition######################
        motion_style = None
        music_style = None
        input_n = self.args.input_n
        output_n = self.args.output_n

        if self.use_music:
            music_style = music

        if self.use_style:
            style_src_pos = torch.tensor([i for i in range(40)]).unsqueeze(0).cuda()
            style_pos_embed = self.position_enc(style_src_pos)
            style = self.src_emb(style) + style_pos_embed

            for enc_layer in self.style_stack:
                style, _ = enc_layer(style, condition=None)
            style = style[:, 0, :]
            motion_style = style
        ###################################################

        src_pos = torch.tensor([i for i in range(input_n+output_n)]
                               ).unsqueeze(0).cuda()
        pos_embed = self.position_enc(src_pos)
        src_pad_idx = torch.tensor([0 for _ in range(input_n)] + [1 for _ in range(output_n)]
                               ).unsqueeze(0).cuda()
        pad_embed = self.padding_embedding(src_pad_idx)
        beat_embed = self.beat_position_enc(beat)
        enc_output = self.src_emb(x) + pos_embed + pad_embed

        for i, beat_enc_layer in enumerate(self.beat_stack):
            beat_embed, _ = beat_enc_layer(enc_output, condition=None)

        for i, enc_layer in enumerate(self.layer_stack):
            if self.use_music:
                enc_output, enc_slf_attn = enc_layer(enc_output, condition=music_style+motion_style)
            else:
                enc_output, enc_slf_attn = enc_layer(enc_output, condition=motion_style)
            if self.use_hist and i == 2:
                enc_output = self.mag(enc_output, hist, beat_embed)

        pose_result = self.pose_decode(enc_output)
        return pose_result[:, -output_n:, :]

    def forward(self, targets, **kwargs):
        style_input = kwargs["styles"]
        contexts_input = kwargs["contexts"]
        beats = kwargs["beats"]

        input_n = self.args.input_n
        output_n = self.args.output_n
        self.kernel_size = 10
        history_n = self.args.input_n

        if self.use_hist:
            src_tmp = contexts_input.clone()
            bs = contexts_input.shape[0]
            src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(history_n - output_n)].clone()
            src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

            vn = history_n - self.kernel_size - output_n + 1
            vl = self.kernel_size + output_n
            idx = np.expand_dims(np.arange(vl), axis=0) + \
                  np.expand_dims(np.arange(vn), axis=1)
            src_value_tmp = src_tmp[:, idx].clone().reshape(
                [bs * vn, vl, -1])
            src_value_tmp = src_value_tmp.reshape([bs, vn, vl, -1])[:, :, -output_n:, :]
            att_pad = torch.zeros(bs, vn, input_n, self.args.node_n).float().cuda()
            src_value_tmp = torch.cat([att_pad, src_value_tmp], dim=2)
            src_value_tmp = self.src_emb(src_value_tmp)
            key_tmp = self.convK(src_key_tmp)
            query_tmp = self.convQ(src_query_tmp)
            score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15
            att_tmp = self.softmax(score_tmp).squeeze(1)
            att_tmp = torch.einsum('bx,bxtd->btd', att_tmp, src_value_tmp)
        else:
            att_tmp = None

        total_loss = {}
        i = 0
        target_pose = targets[:, i + input_n:i + input_n + output_n, :]
        beat_input_length = input_n + output_n
        beat_input = beats[:, i:i + beat_input_length]
        input_pose = targets[:, i:i + input_n, :]

        style_music = kwargs["style_music"]
        same_style_music = kwargs["same_style_music"]
        other_style_music = kwargs["other_style_music"]

        if self.use_music:
            style_music = self.music_linear(self.music_encoder(style_music)[:, 0, :])
            same_style_music = self.music_linear(self.music_encoder(same_style_music)[:, 0, :])
            other_style_music = self.music_linear(self.music_encoder(other_style_music)[:, 0, :])

        input_pose_padding = self.input_padding(input_pose)
        pred_pose = self.run(input_pose_padding, music=style_music, hist=att_tmp, style=style_input, beat=beat_input)

        loss = self.compute_loss(pred_pose, target_pose, style_music, same_style_music, other_style_music)
        for k, v in loss.items():
            if k not in total_loss.keys():
                total_loss[k] = v
            else:
                total_loss[k] += v

        return total_loss

    def compute_loss(self, pred_pose, target_pose, style_music, same_style_music, other_style_music):
        l2_loss = self.l2loss(pred_pose[:, :, 3:-2], target_pose[:, :, 3:-2])
        root_loss = self.l2loss(pred_pose[:, :, :3], target_pose[:, :, :3])
        contact_loss = self.l2loss(pred_pose[:, :, -2:], target_pose[:, :, -2:])
        if self.use_music:
            triplet_loss = self.triplet_loss(style_music, same_style_music, other_style_music)
        else:
            triplet_loss = 0.
        loss = l2_loss + root_loss*0.2 + contact_loss*0.02 + triplet_loss*0.1
        return_dict = {"l2_loss": l2_loss, "root_loss": root_loss,
                        "contact_loss": contact_loss, "loss": loss, }
        if self.use_music:
            return_dict["triplet_loss"] = triplet_loss
        return return_dict

    def input_padding(self, input_pose):
        i_idx = [self.args.input_n - 1 for _ in range(self.args.output_n)]
        padding = input_pose[:, i_idx, :]
        # set delta value to be zero, no need to move around
        input_with_padding = torch.cat([input_pose, padding], dim=1)
        return input_with_padding

    def generate_old(self, targets, beats, music, style):
        """
        :param x: [input_n, node]
        :param music: [t_all, dim_music]
        :return:
        """
        input_n = self.args.input_n
        music_total_length = music.shape[1]
        music_input_length = self.args.input_n

        input_n = self.args.input_n
        history_n = self.args.input_n
        output_n = self.args.output_n
        beat_input_length = input_n + output_n
        if self.use_hist:
            self.kernel_size = 10

            src_tmp = targets.clone()
            # src_tmp[:, :, 19*3:] *= 0.  # 把root信息mask掉，只提供动作的信息
            bs = targets.shape[0]
            src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(history_n - output_n)].clone()
            src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

            vn = history_n - self.kernel_size - output_n + 1
            vl = self.kernel_size + output_n
            idx = np.expand_dims(np.arange(vl), axis=0) + \
                  np.expand_dims(np.arange(vn), axis=1)
            src_value_tmp = src_tmp[:, idx].clone().reshape(
                [bs * vn, vl, -1])
            src_value_tmp = src_value_tmp.reshape([bs, vn, vl, -1])[:, :, -output_n:, :]
            att_pad = torch.zeros(bs, vn, input_n, self.args.node_n).float().cuda()
            src_value_tmp = torch.cat([att_pad, src_value_tmp], dim=2)
            src_value_tmp = self.src_emb(src_value_tmp)
            key_tmp = self.convK(src_key_tmp)

        total_cir = music_total_length - music_input_length + 1
        output_pose = targets[:, :input_n, :]
        for i in range(total_cir):
            if self.use_hist:
                query_tmp = self.convQ(src_query_tmp)
                score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15
                att_tmp = self.softmax(score_tmp).squeeze(1)
                att_tmp = torch.einsum('bx,bxtd->btd', att_tmp, src_value_tmp)
            else:
                att_tmp = None

            music_input = music[:, i:i + music_input_length, :]

            beat_input = beats[:, i:i + beat_input_length]
            if self.use_music:
                style_music = self.music_linear(self.music_encoder(music_input)[:, 0, :])
            else:
                style_music = None
            if i == 0:
                input_pose = targets[:, i:i+input_n, :]
            else:
                input_pose = torch.cat([input_pose[:, 1:, :], pred_pose[:, :1, :]], 1)

            input_pose_padding = self.input_padding(input_pose).detach()
            pred_pose = self.run(input_pose_padding, music=style_music,
                                 style=style.detach(),
                                 hist=att_tmp,
                                 beat=beat_input.detach())
            # 需要加detach，否则会out of memory
            pred_pose = pred_pose.detach()
            output_pose = torch.cat([output_pose, pred_pose[:, :1, :]], 1)

            if total_cir > 0 and self.use_hist:
                # update query
                src_query_tmp = output_pose.transpose(1, 2)[:, :, -self.kernel_size:].clone()

                # update key
                history_n += 1
                src_key_tmp = output_pose.transpose(1, 2)[:, :, :(history_n - output_n)].clone()
                key_tmp = self.convK(src_key_tmp)

                # update value
                value_added = output_pose[:, -output_n:, :].unsqueeze(1)
                att_pad_added = torch.zeros(bs, 1, input_n, self.args.node_n).float().cuda()
                src_value_tmp_added = torch.cat([att_pad_added, value_added], dim=2)
                src_value_tmp_added = self.src_emb(src_value_tmp_added)
                src_value_tmp = torch.cat([src_value_tmp, src_value_tmp_added], dim=1)

        return output_pose[:, :, :-2]