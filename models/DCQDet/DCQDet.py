import math
import copy
import os
import time

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .backbone import build_backbone
from .fpn import build_FPN
from models.CenterGeneration.center_generation import build_center_generation
from .dynamic_decoder import build_decoder
from .matcher import build_matcher

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from util.center_loss import FocalLoss

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

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

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
                Parameters:
                    outputs: raw outputs of the model
                    target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                                  For evaluation, this must be the original image size (before any data augmentation)
                                  For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2] #todo:waht???
        labels = topk_indexes % out_logits.shape[2] #todo
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
        3) we compute the loss between gt_heatmap & heatmap
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, mse_loss=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.hm_crit = torch.nn.MSELoss() if mse_loss else FocalLoss()

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        # (bs, num_queries * num_scales, num_classes)
        src_logits = outputs['pred_logits']

        # (batch_indices, query_indices)
        idx = self._get_src_permutation_idx(indices)
        # (bs*num_matched,)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # (bs, num_queries * num_scales)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):

        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def pad_hm(self, tgt_hm, batch_shape):
        dtype = tgt_hm.dtype
        device = tgt_hm.device
        heatmap = torch.zeros(batch_shape,dtype=dtype, device=device)
        heatmap[:tgt_hm.shape[0], :tgt_hm.shape[1], :tgt_hm.shape[2]].copy_(tgt_hm)
        return heatmap

    def loss_heatmap(self, outputs, targets, indices, num_boxes):
        assert 'heatmap' in outputs
        outputs['heatmap'] = _sigmoid(outputs['heatmap'])
        batch_shape = outputs['heatmap'][0].shape

        losses = {}
        hm_loss = 0

        target_hm = []
        for t in targets:
            t['heatmap'] = self.pad_hm(t['heatmap'], batch_shape)
            target_hm.append(t['heatmap'])
        target_hm = torch.stack(target_hm, dim=0)

        hm_loss = 0
        hm_loss += self.hm_crit(outputs['heatmap'], target_hm)
        losses['heatmap'] = hm_loss
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'heatmap': self.loss_heatmap,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
        The expected keys in each dict depends on the losses applied
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        wh = outputs_without_aux['pred_wh']
        indices = self.matcher(outputs_without_aux, targets)
        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        # ['labels', 'boxes', 'heatmap', 'cardinality']
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    indices = self.matcher(aux_outputs, targets)
                    for loss in self.losses:
                        if loss == 'masks' or loss== 'heatmap':
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs['log'] = False
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
        return losses

class DCQ_Det(nn.Module):
    """Performs object detection"""
    def __init__(self,
                 backbone,
                 FPN,
                 center_generation,
                 decoder,
                 num_classes,
                 num_pos_queries,
                 num_scales=5,
                 aux_loss=True,
                 heatmap_scale=8):
        """ Initializes the model.
                Parameters:
                    backbone: torch module of the backbone to be used. See backbone.py
                    transformer: torch module of the transformer architecture. See transformer.py
                    num_classes: number of object classes
                    num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                                 Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
                    aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
         """
        super().__init__()
        self.backbone = backbone
        self.FPN = FPN
        self.center_generation = center_generation
        self.decoder = decoder
        self.num_pos_queries = num_pos_queries
        self.num_scales = num_scales
        self.aux_loss = aux_loss
        self.heatmap_scale = heatmap_scale
        self.num_classes = num_classes

        hidden_dim = decoder.d_model
        # Prediction Heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.wh_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.bbox_embed = self.decoder.bbox_embed
        self.bbox_embed_type = self.decoder.bbox_embed_type
        if self.bbox_embed_type == 'diff_each_layer':
            for bbox_embedx in self.bbox_embed:
                nn.init.constant_(bbox_embedx.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embedx.layers[-1].bias.data, 0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.wh_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.wh_embed.layers[-1].bias.data, 0)

        self.num_pred = self.decoder.num_layers
        #self.wh_embed = _get_clones(self.wh_embed, self.num_pred)
        self.decoder.class_embed = self.class_embed
        self.decoder.wh_embed = self.wh_embed

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_h, valid_ratio_w], -1)
        return valid_ratio

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_cls": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_center"
               - "pred_wh"
               - "pred_heatmap"
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        BS, _, H, W = samples.tensors.shape
        # TODO :feat_pos_embeds: bs, hidden_dim or hidden_dim//2 ???, h, w
        feat, feat_pos_embeds = self.backbone(samples)
        # feat:[p2, p3, p4, p5]; p_level: bs, c, h, w
        # feat_masks:[m2, m3, m4, m5], m_level: bs, h, w
        feat, feat_masks = self.FPN(feat)
        # valid_ratios: bs, scale, 2(h, w)
        # 终于写出来代码了，舒服😌～
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in feat_masks], 1)


        # 怎么这么多bug😭😭
        heatmap, refpoints = self.center_generation(feature=feat[0],
                                                    mask=feat_masks[0],
                                                    valid_ratio=valid_ratios[:, 0, :],
                                                    top_K=self.num_pos_queries,
                                                    out_height=H,
                                                    out_width=W)
        # feat: List([bs, c, h, w]) equal to src
        # reference_points: [bs, top_K, 2(x, y)], range[0, 1], used to calculate content&position query
        # hs: (num_layers, bs, num_queries, scales, c), reference: (num_layers, bs, num_queries, scales, 2)
        hs, reference= self.decoder(reference_points=refpoints, feat=feat, valid_ratios=valid_ratios,
                                    pos=feat_pos_embeds, feature_key_padding_mask=feat_masks)

        """Prediction"""
        centers = []
        reference_before_sigmoid = inverse_sigmoid(reference)
        assert reference.shape[-1] == 2
        # hack implementationfor self.bbox_embed_type != 'diff_each_layer'
        for lvl in range(hs.shape[0]):
            tmp = self.bbox_embed[lvl](hs[lvl])
            tmp += reference_before_sigmoid[lvl]
            center = tmp.sigmoid()  # bs, num_queries, scales, 2
            centers.append(center)
        # num_layers, bs, num_queries * scales, 2
        centers = torch.stack(centers).view(self.num_pred, BS, -1, 2)

        # num_layers, bs, num_queries * scales, num_classes
        classes = self.class_embed(hs).view(self.num_pred, BS, -1, self.num_classes)
        # num_layers, bs, num_queries * scales, 2(w,h)
        wh = self.wh_embed(hs).view(self.num_pred, BS, -1, 2).sigmoid()

        bboxes = torch.cat([centers, wh], dim=-1)  # num_layers, bs, num_queries * scales, 4(cx,cy,w,h)
        # TODO: calculate loss of centers, wh, classes instead of boxes, classes
        out = {'pred_logits': classes[-1], 'pred_boxes': bboxes[-1], 'pred_centers': centers[-1], 'pred_wh': wh[-1], 'heatmap': heatmap}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(classes, bboxes)
        return out

def build_DCQ_Det(args):

    if args.dataset_file == 'kitti':
        num_classes = 4
    elif args.dataset_file == 'coco':
        num_classes = 91
    else:
        raise NotImplementedError("Unknown args.dataset_file: {}".format(args.dataset_file))

    device = torch.device(args.device)
    backbone = build_backbone(args)
    FPN = build_FPN(args)
    center_generation = build_center_generation(args)
    decoder = build_decoder(args)

    model = DCQ_Det(
        backbone,
        FPN,
        center_generation,
        decoder,
        num_classes=num_classes,
        num_pos_queries=args.num_pos_queries,
        num_scales=args.num_scales,
        aux_loss=args.aux_loss,
        heatmap_scale=args.heatmap_scale
    )

    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_hm': args.hm_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'heatmap', 'cardinality']
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha, mse_loss=args.mse_loss)

    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    return model, criterion, postprocessors