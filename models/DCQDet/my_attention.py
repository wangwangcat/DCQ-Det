import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import MultiScaleDeformableAttention as MSDA


class SingleScaleDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=2):
        super().__init__()
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_padding_mask=None):
        """
        query: bs, num_query, feature_levels, channel  # query = position_query + content_query
        reference_points: bs, num_query, feature_levels, 2 or 4 (range in [0, 1])
        input_flatten: [p3, p4, p5], p_level: bs, h_level * w_level, channel
        input_spatial_shapes: feature_levels, 2
        input_padding_mask: [m3, m4, m5], m_level: bs, h_level * w_level, 1

        -return output:bs, num_query, feature_levels, channel
        """
        bs, num_queries, feature_levels, _ = query.shape
        for level, feature_map in enumerate(input_flatten):
            bs, len_in, _ = input_flatten.shape
            assert input_spatial_shapes[level, 0] * input_spatial_shapes[level, 1] == len_in

            value = self.value_proj(input_flatten)
            if input_padding_mask is not None:
                value = value.masked_fill(input_padding_mask[level][..., None], float(0))
            value = value.view(bs, len_in, self.n_heads, self.d_model // self.n_heads)

            # self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
            sampling_offsets = self.sampling_offsets(query).view(bs, num_queries, feature_levels, self.n_heads,
                                                                 self.n_levels, self.n_points, 2)
            attention_weights = self.attention_weights(query).view(bs, num_queries, feature_levels, self.n_heads,
                                                                   self.n_levels * self.n_points)
            attention_weights = F.softmax(attention_weights, -1).view(bs, num_queries, feature_levels, self.n_heads,
                                                                      self.n_levels, self.n_points)

class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()
