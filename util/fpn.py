from typing import Text, Tuple, Union
import math
import torch
from torch import nn
import torch.nn.functional as F

def parse_image_size(image_size: Union[Text, int, Tuple[int, int]]):
    """Parse the image size and return (height, width).
    Args:
      image_size: A integer, a tuple (H, W), or a string with HxW format.
    Returns:
      A tuple of integer (height, width).
    """
    if isinstance(image_size, int):
      # image_size is integer, with the same width and height.
      return (image_size, image_size)

    if isinstance(image_size, str):
      # image_size is a string with format WxH
      width, height = image_size.lower().split('x')
      return (int(height), int(width))

    if isinstance(image_size, tuple):
      return image_size

    raise ValueError('image_size must be an int, WxH string, or (height, width)'
                     'tuple. Was %r' % image_size)


def get_feat_sizes(image_size: Union[Text, int, Tuple[int, int]], max_level: int):
    """
    Get feature w & h for all the levels
    Args:
      image_size: A integer, a tuple (H, W), or a string with HxW format.
      max_level: maximum feature levels
    Returns:
      feature_sizes:  a list of tuples (height, width) for each level.
    """
    image_size = parse_image_size(image_size)
    # 我要努力写代码，给舍友买大房子，好累啊😭😭
    feature_sizes = [{'height': image_size[0], 'width': image_size[1]}]
    feature_size = image_size
    for _ in range(1, max_level + 1):
        # stride = 2 so spatial dimension reduce
        feature_size = ((feature_size[0] - 1) // 2 + 1, (feature_size[1] - 1) // 2 + 1)
        feature_sizes.append({'height': feature_size[0], 'width': feature_size[1]})
    return feature_sizes


class Conv2dStaticSamePadding(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x


class MaxPool2dStaticSamePadding(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)