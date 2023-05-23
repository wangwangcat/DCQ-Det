import torch
import torch.nn as nn
import torch.nn.functional as F

from util.fpn import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding
from util.fpn import MemoryEfficientSwish, Swish

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class FPN(nn.Module):
    def __init__(self, num_channels, conv_channels, epsilon=1e-4, onnx_export=False, attention=True):
        """
        Args:
            num_channels: you know
            conv_channels: you know too
            first_time: whether the input comes directly from the efficientnet,(of course not)
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(FPN, self).__init__()
        self.epsilon = epsilon
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        # Conv layers
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv2_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)

        # Generate P6 & P7
        self.p5_to_p6 = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[3], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            MaxPool2dStaticSamePadding(3, 2)
        )
        self.p6_to_p7 = nn.Sequential(
            MaxPool2dStaticSamePadding(3, 2)
        )

        # Down Channel
        self.p2_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
        )
        self.p3_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
        )
        self.p4_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
        )
        self.p5_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[3], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
        )
        self.p4_down_channel_2 = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
        )
        self.p5_down_channel_2 = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
        )

        # Attention Weight
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.p2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_w1_relu = nn.ReLU()

        self.p3_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w2_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, feature_maps):
        """With Self-Attention"""
        p2 = feature_maps[0].tensors
        p3 = feature_maps[1].tensors
        p4 = feature_maps[2].tensors
        p5 = feature_maps[3].tensors

        # Put in the same channel
        p6_in = self.p5_to_p6(p5)
        p7_in = self.p6_to_p7(p6_in)
        p2_in = self.p2_down_channel(p2)
        p3_in = self.p3_down_channel(p3)
        p4_in = self.p4_down_channel(p4)
        p5_in = self.p5_down_channel(p5)

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * F.interpolate(p7_in, size=p6_in.shape[-2:])))
        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * F.interpolate(p6_up, size=p5_in.shape[-2:])))
        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * F.interpolate(p5_up, size=p4_in.shape[-2:])))
        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_up = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * F.interpolate(p4_up, size=p3_in.shape[-2:])))
        # Weights for P2_0 and P3_1 to P2_2
        p2_w1 = self.p2_w1_relu(self.p2_w1)
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        p2_out = self.conv2_up(self.swish(weight[0] * p2_in + weight[1] * F.interpolate(p3_up, size=p2_in.shape[-2:])))

        # Weights for P3_0, P3_1 and P2_2 to P3_2
        p3_w2 = self.p3_w2_relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        p3_out = self.conv3_down(
            self.swish(weight[0] * p3_in + weight[1] * p3_up + weight[2] * self.p4_downsample(p2_out)))
        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))
        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))
        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        feat = [p2_out, p3_out, p4_out, p5_out]
        masks = [feature_maps[0].mask, feature_maps[1].mask, feature_maps[2].mask, feature_maps[3].mask]
        return feat, masks


def build_FPN(args):
    return FPN(args.num_channels, args.conv_channels, epsilon=1e-4, onnx_export=True, attention=True)