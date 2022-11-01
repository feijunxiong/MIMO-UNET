from mindspore import nn
from src.init_weights import KaimingUniform, UniformBias


class Conv2dLikeTorch(nn.Conv2d):
    """Conv2dTransposeLikeTorch"""
    def __init__(self, in_channel, out_channel, kernel_size, pad_mode, padding, stride, has_bias):
        initializer = KaimingUniform()
        bias_initializer = UniformBias(shape=(out_channel, in_channel, kernel_size, kernel_size), mode='fan_in')
        super().__init__(in_channel, out_channel, kernel_size, pad_mode=pad_mode, weight_init=initializer,
                         padding=padding, stride=stride, has_bias=has_bias, bias_init=bias_initializer)


class Conv2dTransposeLikeTorch(nn.Conv2dTranspose):
    """Conv2dTransposeLikeTorch"""
    def __init__(self, in_channel, out_channel, kernel_size, pad_mode, padding, stride, has_bias):
        initializer = KaimingUniform(mode='fan_in')
        bias_initializer = UniformBias(shape=(out_channel, in_channel, kernel_size, kernel_size), mode='fan_in')
        super().__init__(in_channel, out_channel, kernel_size, pad_mode=pad_mode, weight_init=initializer,
                         padding=padding, stride=stride, has_bias=has_bias, bias_init=bias_initializer)


class BasicConv(nn.Cell):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super().__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                Conv2dTransposeLikeTorch(in_channel, out_channel, kernel_size, pad_mode='pad',
                                         padding=padding, stride=stride, has_bias=bias)
            )
        else:
            layers.append(
                Conv2dLikeTorch(in_channel, out_channel, kernel_size, pad_mode='pad',
                                padding=padding, stride=stride, has_bias=bias)
            )
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU())
        self.main = nn.SequentialCell(layers)

    def construct(self, x):
        return self.main(x)


class ResBlock(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.main = nn.SequentialCell(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def construct(self, x):
        return self.main(x) + x
