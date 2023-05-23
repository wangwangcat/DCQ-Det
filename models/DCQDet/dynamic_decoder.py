import math
import copy
import os
from typing import Optional, List
import time

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .attention import MultiheadAttention
from util.misc import inverse_sigmoid
from ..CenterGeneration.center_generation import gather_feat

def gen_sineembed_for_position(pos_tensor):
    # n_query, scale, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, scale, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, :, 0] * scale
    y_embed = pos_tensor[:, :, :, 1] * scale
    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

    if len(pos_x.shape) != 4:
        pos_x = pos_x.unsqueeze(2)
        pos_y = pos_y.unsqueeze(2)

    pos = torch.cat((pos_y, pos_x), dim=3)
    return pos


class DynamicDecoder(nn.Module):
    def __init__(self, num_layers, return_intermediate=False,
                 d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                 modulate_hw_attn=True,
                 iter_update=True, bbox_embed_type='fix', num_scales=3,   # for iter update coord
                 no_sine_embed=False,
                 nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",  num_queries=100,
                 ):
        super().__init__()
        decoder_layer = DynamicDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                            dropout=dropout, activation=activation, keep_query_pos=keep_query_pos,
                                            num_scale=num_scales, num_queries=num_queries)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim
        self.no_sine_embed = no_sine_embed

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        # query_scale_type is cond_elewise
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))

        if self.no_sine_embed:
            self.ref_point_head = MLP(query_dim, d_model, d_model, 3)
        else:
            self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)

        self.num_scales = num_scales
        self.iter_update = iter_update
        assert bbox_embed_type in ['diff_each_layer', 'diff_each_scale', 'fix']
        self.bbox_embed_type = bbox_embed_type
        if iter_update:
            if bbox_embed_type == 'diff_each_layer':
                # input_dim, hidden_dim, output_dim, num_layers
                self.bbox_embed = nn.ModuleList([MLP(d_model, d_model, 2, 3) for _ in range(6)])
            elif bbox_embed_type == 'diff_each_scale':
                self.bbox_embed = nn.ModuleList([MLP(d_model, d_model, 2, 3) for _ in range(self.num_scales)])
            else:
                self.bbox_embed = MLP(d_model, d_model, 2, 3)

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn

        if modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)

        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(self,
                reference_points, # position in real feature map(no padding) ranging[0,1]: bs, top_k, 2(x, y)
                feat, # [p3, p4, p5, p6], p_level: bs, c, h, w
                valid_ratios, # bs, scale, 2(h, w)
                pos: Optional[Tensor] = None, # for cross attn key postion
                tgt_mask: Optional[Tensor] = None,
                feature_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                feature_key_padding_mask: Optional[Tensor] = None,
                 ):

        reference_points = reference_points.transpose(0, 1)   # num_queries, bs, 2(x, y)
        ms_content = []
        feat_flatten = [] # [p3, p4, p5]
        pos_flatten = [] # [p3, p4, p5], the pos embedding for feature map
        ms_refpoints_unsigmoid = []
        for level, (feature_map, feat_pos) in enumerate(zip(feat, pos)):
            if level == 0:
                continue
            h, w  = feature_map.shape[-2:]
            valid_h = valid_ratios[:, level, 0] * h
            valid_w = valid_ratios[:, level, 1] * w
            feature_map_flatten = feature_map.flatten(2).permute(2, 0, 1) # h * w, bs, c
            feat_pos_flatten = feat_pos.flatten(2).permute(2, 0, 1)

            # TODO: refpoints到底代表什么位置，可能还需要修改
            refpoints = torch.stack([reference_points[:, :, 0] * valid_w, reference_points[:, :, 1] * valid_h], dim=2)
            refpoints = refpoints.to(torch.int64)   # num_queries, bs, 2(x, y)
            # TODO: may be mistake
            index = refpoints[:, :, 0] + refpoints[:, :, 1] * w  # num_queries, bs
            content = gather_feat(feature_map_flatten.transpose(0, 1), index.transpose(0, 1)).transpose(0, 1)   # num_queries, bs, c
            ms_content.append(content)
            feat_flatten.append(feature_map_flatten)
            pos_flatten.append(feat_pos_flatten)
            ms_refpoints_unsigmoid.append(refpoints)

        ms_content = torch.stack(ms_content, dim=1)   # num_queries, scales, bs, c
        ms_refpoints_unsigmoid = torch.stack(ms_refpoints_unsigmoid, dim=1)   # num_queries, scales, bs, 2(x, y)

        output = ms_content
        intermediate = []
        ms_refpoints = ms_refpoints_unsigmoid.sigmoid()
        all_refpoints = [ms_refpoints]

        for layer_id, layer in enumerate(self.layers):
            # Deformable DETR
            assert ms_refpoints.shape[-1] == 2

            """Get Position Embedding for SA-Query"""
            # PE(A_q) = Cat(PE(x_q), PE(y_q))
            query_sine_embed = gen_sineembed_for_position(ms_refpoints)  # num_queries, scales, bs, 256
            # Anchor A_q's pisitional query: P_q = MLP(PE(A_q))
            # self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2), query_dim=2
            query_pos = self.ref_point_head(query_sine_embed)  # num_queries, scales, bs, 256

            """To transform refpoints position embeddings"""
            if self.query_scale_type != 'fix_elewise':
                pos_transformation = self.query_scale(output)
                #if layer_id == 0:
                #    pos_transformation = 1
                #else:
                #    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]
            query_sine_embed = query_sine_embed[..., :self.d_model] * pos_transformation

            """Modulated HW attentions"""
            if self.modulate_hw_attn:
                # Calculate w&h for pos attn based on output(content queries)
                refHW_cond = self.ref_anchor_head(output).sigmoid()  # num_queries, scales, bs, 2
                query_sine_embed[..., self.d_model // 2:] *= refHW_cond[..., 0].unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= refHW_cond[..., 1].unsqueeze(-1)

            output = layer(ms_tgt=output,
                           tgt_mask=tgt_mask,
                           feat_flatten=feat_flatten,
                           feature_mask=feature_mask,
                           query_pos=query_pos, # sa, q
                           pos=pos_flatten, # ca, k
                           query_sine_embed=query_sine_embed, # ca, q
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           feature_key_padding_mask=feature_key_padding_mask,
                           ms_ref_points=ms_refpoints)

            """Iterative update"""
            # hack implementation for diff_each_scale
            if self.iter_update:
                if self.bbox_embed_type == 'diff_each_layer':
                    tmp = self.bbox_embed[layer_id](output)
                else:
                    tmp = self.bbox_embed(output)
                # import ipdb; ipdb.set_trace()
                tmp[..., :self.query_dim] += inverse_sigmoid(ms_refpoints)
                new_ms_refpoints = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    all_refpoints.append(new_ms_refpoints)
                ms_refpoints = new_ms_refpoints.detach()

            """Intermediate output"""
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    # num_layers, bs, num_queries, scales, c
                    torch.stack(intermediate).transpose(1, 2).permute(0, 3, 2, 1, 4),
                    torch.stack(all_refpoints).transpose(1, 2).permute(0, 3, 2, 1, 4)
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2).permute(0, 3, 1, 2, 4),
                    ms_refpoints.unsqueeze(0).transpose(1, 2).permute(0, 3, 1, 2, 4)
                ]

        return output.unsqueeze(0)


class DynamicDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", keep_query_pos=False, num_scale=3, num_queries=100):
        super().__init__()
        self.num_scale = num_scale
        self.num_queries = num_queries
        # For Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)

        #self.self_attn_scale = nn.ModuleList(MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model) for _ in range(self.num_scale))
        self.cross_attn_scale = nn.ModuleList(MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model) for _ in range(self.num_scale))
        #self.self_attn_query =MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        #self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)
        self.nhead = nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.keep_query_pos = keep_query_pos

    def forward(self,
                ms_tgt,  # num_queries, scales, bs, 2
                tgt_mask,
                feat_flatten,  # [p3, p4, p5], p_level: h * w, bs, c
                feature_mask,
                query_pos,  # sa, q, # num_queries, scales, bs, 256
                pos,  # ca, k, [p3, p4, p5], p_level: bs, c, h, w --> h * w, bs, c
                query_sine_embed,  # ca, q  # num_queries, scales, bs, 2
                tgt_key_padding_mask,
                feature_key_padding_mask,   # [m2, m3, m4, m5], m_level: bs, h, w
                ms_ref_points  # bs, 3, K, 2
                ):
        tgt_out = []
        # multiple scales

        num_queties, scale, bs, c = ms_tgt.shape
        # ======Self-Attention Begin======
        tgt = ms_tgt.view(-1, bs, c)  # num_queties, scale, bs, c
        query_pos_sa = query_pos.view(-1, bs, c)
        sa_q_pos = self.sa_qpos_proj(query_pos_sa)
        sa_k_pos = self.sa_kpos_proj(query_pos_sa)
        sa_q_content = self.sa_qcontent_proj(tgt)
        sa_k_content = self.sa_kcontent_proj(tgt)
        q = sa_q_content + sa_q_pos  # num_queries, bs, n_model
        k = sa_k_content + sa_k_pos
        v = self.sa_v_proj(tgt)

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                         key_padding_mask=tgt_key_padding_mask)[0]
        # ======Self-Attention End======

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = tgt.view(num_queties, scale, bs, c)

        for i, cross_attn in enumerate(self.cross_attn_scale):
            """# ======Self-Attention Begin======
            ref_points = ms_ref_points[:, i, :, :]
            tgt = ms_tgt[:, i, :, :]  # num_queties, bs, c
            query_pos_lvl = query_pos[:, i, :, :]

            sa_q_pos = self.sa_qpos_proj(query_pos_lvl)
            sa_k_pos = self.sa_kpos_proj(query_pos_lvl)
            sa_q_content = self.sa_qcontent_proj(tgt)
            sa_k_content = self.sa_kcontent_proj(tgt)
            q = sa_q_content + sa_q_pos   # num_queries, bs, n_model
            k = sa_k_content + sa_k_pos
            v = self.sa_v_proj(tgt)

            tgt2 = self_attn(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            # ======Self-Attention End======

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)"""

            # ======Cross-Attention Begin======
            # For cross-attention: Q_q = Cat(C_q, PE(x_q, y_q), MLP(C_q))
            # K_x,y = Cat(F_x,y, PE(x, y)), V_x,y = F_x,y
            pos_lvl = pos[i]
            tgt_lvl = tgt[:, i, :, :]
            query_pos_lvl = query_pos[:, i, :, :]

            feature_map = feat_flatten[i]   # h * w, bs, c
            ca_q_content = self.ca_qcontent_proj(tgt_lvl)
            ca_k_content = self.ca_kcontent_proj(feature_map)
            v = self.ca_v_proj(feature_map)

            num_queries, bs, n_model = ca_q_content.shape
            hw, _, _ = ca_k_content.shape
            k_pos = self.ca_kpos_proj(pos_lvl)

            if self.keep_query_pos:
                ca_q_pos = self.ca_qpos_proj(query_pos_lvl)
                q = ca_q_content + ca_q_pos
                k = ca_k_content + k_pos
            else:
                q = ca_q_content
                k = ca_k_content

            q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
            query_sine_embed_lvl = self.ca_qpos_sine_proj(query_sine_embed[:, i, :, :])
            # 100, 4, 8, 32
            query_sine_embed_lvl = query_sine_embed_lvl.view(num_queries, bs, self.nhead, n_model // self.nhead)
            q = torch.cat([q, query_sine_embed_lvl], dim=3).view(num_queries, bs, n_model * 2)

            k = k.view(hw, bs, self.nhead, n_model // self.nhead)
            k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
            k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

            mask = feature_key_padding_mask[i + 1].flatten(1)

            tgt2 = cross_attn(query=q,
                                   key=k,
                                   value=v, attn_mask=feature_mask,
                                   key_padding_mask=mask)[0]
            # ======Cross-Attention End======

            tgt_lvl = tgt_lvl + self.dropout2(tgt2)
            tgt_lvl = self.norm2(tgt_lvl)
            tgt_out.append(tgt_lvl)

        tgt_out = torch.stack(tgt_out, dim=1)   # num_queries, num_scales, bs, c
        """for i in range(self.num_queries):
            # ======Self-Attention Begin======
            tgt = tgt_out[i, :, :, :].clone()  # num_scales, bs, c
            query_pos_lvl = query_pos[i, :, :, :]
            sa_q_pos = self.sa_qpos_proj(query_pos_lvl)
            sa_k_pos = self.sa_kpos_proj(query_pos_lvl)
            sa_q_content = self.sa_qcontent_proj(tgt)
            sa_k_content = self.sa_kcontent_proj(tgt)
            q = sa_q_content + sa_q_pos
            k = sa_k_content + sa_k_pos
            v = self.sa_v_proj(tgt)

            tgt2 = self.self_attn_query(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt_out[i, :, :, :] = self.norm1(tgt)
            # ======Self-Attention End======"""

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt_out))))
        tgt_out = tgt_out + self.dropout3(tgt2)
        tgt_out = self.norm3(tgt_out)

        return tgt_out



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_decoder(args):
    return DynamicDecoder(
        num_layers=args.dec_layers,
        return_intermediate=True,
        d_model=args.hidden_dim,
        query_dim=args.query_dim,
        iter_update=True,
        bbox_embed_type=args.bbox_embed_type,
        num_scales=args.num_scales,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.transformer_activation,
        num_queries=args.num_pos_queries
    )