#try:
#    from nms import soft_nms
#except:
#    print('NMS not imported! If you need it,'
#          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')

import numpy as np
import torch
from torch import nn
import cv2


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

# trans = get_affine_transform(center[i], scale, 0, out_size, inv=1)
def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

# target_coords[k, :] = affine_transform(coords[k, :], trans)
def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]



def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def gather_feat(feat, ind, mask=None):
    """
    feat: topk_ys:[bs, cls*K, 1]
    ind: topk_ind:[bs, K]
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat



def _topk(scores, K=100):
    # scores = heatmap
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs



class CenterGeneration(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.world_size >= 0:
            args.device = torch.device('cuda')
        else:
            args.device = torch.device('cpu')

        print('Creating Center Generation...')
        self.num_channels = args.num_channels
        self.head_conv = args.head_conv
        self.num_heatmap = args.num_heatmap

        self.pred_head = nn.Sequential(
            nn.Conv2d(self.num_channels, self.head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, self.num_heatmap, kernel_size=1, stride=1, padding=0))
        #self.pred_head = load_head(self.pred_head, args.load_head)
        self.pred_head = self.pred_head.to(args.device)
        self.max_per_image = 100
        self.num_classes = args.num_classes
        self.args = args

    def hm_decode(self, heat, top_K=100):
        batch, cat, height, width = heat.size()

        # heat = torch.sigmoid(heat)
        # perform nms on heatmaps
        heat = _nms(heat)

        scores, inds, clses, ys, xs = _topk(heat, K=top_K)
        xs = xs.view(batch, top_K, 1) + 0.5
        ys = ys.view(batch, top_K, 1) + 0.5

        # clses = clses.view(batch, K, 1).float()
        # scores = scores.view(batch, K, 1)
        center_points = torch.cat([xs, ys], dim=2)

        # detections = torch.cat([center_points, scores, clses], dim=2)
        return center_points

    def process(self, feature, top_K, mask):
        with torch.no_grad():
            # feature: [bs, num_channels, w, h]
            heatmap = self.pred_head(feature)
            heatmap = torch.sigmoid(heatmap)

            # Whether to leverage horizontal flipping for test images to improve performance??
            # if self.opt.flip_test:
            #    heatmap = (heatmap[0:1] + flip_tensor(heatmap[1:2])) / 2

            torch.cuda.synchronize()
            if len(mask.shape) < len(heatmap.shape):
                mask = mask.unsqueeze(1)
            heatmap = torch.masked_fill(input=heatmap, mask=mask, value=0)
            center_points = self.hm_decode(heatmap, top_K)

        return heatmap, center_points

    def forward(self, feature, mask, valid_ratio, top_K, out_height, out_width):
        """
        feature = feat[0]: bs, c, h, w
        valid_ratios: bs, scale, 2(h, w)
        top_k == num_quries, representing the number of points
        output:
        refpoints: bs, top_K, 2(x, y), range in [0, 1]
        """

        heatmap, center_points = self.process(feature, top_K, mask)
        torch.cuda.synchronize()
        h, w = feature.shape[-2:]
        refpoints = center_points
        # TODO: May be wrong
        refpoints[:, :, 0] = refpoints[:, :, 0] / (w * valid_ratio[:, None, 1])
        refpoints[:, :, 1] = refpoints[:, :, 1] / (h * valid_ratio[:, None, 0])

        """"# feature: batch, channel, scale, scale
        scale = feature.shape[-1]
        center = (scale / 2, scale / 2)
        out_size = (out_width, out_height)

        # center_points: batch, k, 2
        for i in range(center_points.shape[0]):
            coords = center_points[i, :, :]
            target_coords = np.zeros(coords.shape)
            trans = get_affine_transform(center, scale, 0, out_size, inv=1)
            for k in range(coords.shape[0]):
                target_coords[k, :] = affine_transform(coords[k, :], trans)
            center_points[i, :, :] = target_coords"""

        return heatmap, refpoints

def build_center_generation(args):
    return CenterGeneration(args)
