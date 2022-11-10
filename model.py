import math
from typing import Optional

import torch.nn as nn
from torch import Tensor, save, load


class CorrectDepth:
    def __init__(self, alpha: float, min_depth: int = 3):
        self.alpha = alpha
        self.min_depth = min_depth

    def __call__(self, depth: int) -> int:
        """
            Adjust a depth value
            :param depth: (int) Depth/channels
            :return: (int) corrected depth
        """
        return max(self.min_depth, int(depth * self.alpha))


class BaseLayer(nn.Module):
    @staticmethod
    def choice_nl(nl):
        if nl == 'RE':
            return nn.ReLU()
        elif nl == 'HS':
            return nn.Hardswish()
        elif nl is None:
            return None
        else:
            raise ValueError('nl should be "RE", "HS" or None')

    @staticmethod
    def same_padding(kernel_size):
        return (kernel_size - 1) // 2


class SqueezeAndExcite(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        sqz_channels = math.ceil(channels / 4)
        self.sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, sqz_channels, 1),
            nn.ReLU(),
            nn.Conv2d(sqz_channels, channels, 1),
            nn.Hardsigmoid()
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * self.sequential(inputs)


class DepthWiseConv(BaseLayer):
    def __init__(self, channels: int, kernel_size: int or tuple,
                 non_linear: str, stride: int = 1):
        super().__init__()

        self.depth_wise = nn.Conv2d(channels,
                                    channels,
                                    kernel_size,
                                    stride,
                                    self.same_padding(kernel_size),
                                    groups=channels)

        self.non_linear = self.choice_nl(non_linear)
        self.normalization = nn.BatchNorm2d(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.depth_wise(inputs)
        return self.non_linear(self.normalization(out))


class DepthWiseSepConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int or tuple,
                 squeeze_excite_add: bool, non_linear: str, stride: int = 1):
        super().__init__()
        # add Squeeze and Excitation block
        if squeeze_excite_add:
            self.sae = SqueezeAndExcite(in_channels)
        else:
            self.sae = None

        self.depth_wise_conv = DepthWiseConv(in_channels, kernel_size, non_linear, stride)
        self.point_wise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.depth_wise_conv(inputs)
        if self.sae is not None:
            out = self.sae(out)
        out = self.point_wise(out)
        return out


class BottleNeck(BaseLayer):
    def __init__(self, in_channels: int, exp_channels: int, out_channels: int,
                 kernel_size: int or tuple, squeeze_excite_add: bool,
                 non_linear: str, stride: int = 1):
        super().__init__()
        self.non_linear = self.choice_nl(non_linear)
        self.expansion_layer = nn.Conv2d(in_channels, exp_channels, 1)
        self.depth_wise_sep = \
            DepthWiseSepConv(exp_channels, out_channels, kernel_size,
                             squeeze_excite_add, non_linear, stride)

        self.normalization_bn = nn.BatchNorm2d(exp_channels)
        self.normalization_out = nn.BatchNorm2d(out_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.expansion_layer(inputs)
        out = self.normalization_bn(out)
        out = self.non_linear(out)
        out = self.depth_wise_sep(out)
        out = out + inputs if inputs.size() == out.size() else out
        return self.normalization_out(out)


class Convolution(BaseLayer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int or tuple,
                 batch_norm_add: bool, squeeze_excite_add: bool, non_linear: str, stride: int = 1):
        super().__init__()
        # add Squeeze and Excitation block
        if squeeze_excite_add:
            self.sae = SqueezeAndExcite(out_channels)
        else:
            self.sae = None
        # add batch normalization
        if batch_norm_add:
            self.normalization = nn.BatchNorm2d(out_channels)
        else:
            self.normalization = None

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, self.same_padding(kernel_size))
        self.non_linear = self.choice_nl(non_linear)

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.conv(inputs)
        if self.sae is not None:
            out = self.sae(out)
        if self.normalization is not None:
            out = self.normalization(out)
        if self.non_linear is not None:
            out = self.non_linear(out)
        return out


class MobileNetV3(nn.Module):
    def __init__(self, labels: Optional[list] = None):
        super().__init__()
        self.labels = labels
        self.sequential = None
        self.architecture = None

    def save_labels(self, labels: Optional[list]):
        self.labels = labels

    def choice_model_architecture(
            self, architecture: str, in_channels: int, out_channels: int,
            alpha: float, min_depth: int, dropout: float):
        """
        Return default architecture with users input and output channels
        :param architecture: (str) Architecture should be "large" or "small"
        :param in_channels: (int) input channels in model
        :param out_channels: (int) output channels from model. Classes count
        :param alpha: (float) coefficient of reduce depth size
        :param min_depth: (int) max(min_depth, layer_depth * alpha)
        :param dropout: (float) probability of an element to be zeroed

        BottleNeck: ['bneck', in_ch, exp_ch, out_ch,
                        k_size, sq_exc, non_linear, stride]
        Convolution: ['conv', in_ch, out_ch, k_size,
                        batchnorm, sq_ext, non_linear, stride]
        Pool: ['pool', out_size]
        Dropout: ['dropout', p]
        """
        if type(in_channels) is not int or in_channels < 1:
            raise ValueError('in_channels should be type int and >= 1')
        if type(out_channels) is not int or out_channels < 1:
            raise ValueError('out_channels should be type int and >= 1')
        c_d = CorrectDepth(alpha, min_depth)
        if architecture == 'large':
            self.architecture = (
                # 'conv', in_ch, out_ch, k_size, batchnorm, sq_ext, non_linear, stride
                ['conv', in_channels, c_d(16), 3, True, False, 'HS', 2],  # 224
                # 'bneck', in_ch, exp_ch, out_ch, k_size, sq_exc, non_linear, stride
                ['bneck', c_d(16), c_d(16), c_d(16), 3, False, 'RE', 1],  # 112
                ['bneck', c_d(16), c_d(64), c_d(24), 3, False, 'RE', 2],  # 112
                ['bneck', c_d(24), c_d(72), c_d(24), 3, False, 'RE', 1],  # 56
                ['bneck', c_d(24), c_d(72), c_d(40), 5, True, 'RE', 2],  # 56
                ['bneck', c_d(40), c_d(120), c_d(40), 5, True, 'RE', 1],  # 28
                ['bneck', c_d(40), c_d(120), c_d(40), 5, True, 'RE', 1],  # 28
                ['bneck', c_d(40), c_d(240), c_d(80), 3, False, 'HS', 2],  # 28
                ['bneck', c_d(80), c_d(200), c_d(80), 3, False, 'HS', 1],  # 14
                ['bneck', c_d(80), c_d(184), c_d(80), 3, False, 'HS', 1],  # 14
                ['bneck', c_d(80), c_d(184), c_d(80), 3, False, 'HS', 1],  # 14
                ['bneck', c_d(80), c_d(480), c_d(112), 3, True, 'HS', 1],  # 14
                ['bneck', c_d(112), c_d(672), c_d(112), 3, True, 'HS', 1],  # 14
                ['bneck', c_d(112), c_d(672), c_d(160), 5, True, 'HS', 2],  # 14
                ['bneck', c_d(160), c_d(960), c_d(160), 5, True, 'HS', 1],  # 7
                ['bneck', c_d(160), c_d(960), c_d(160), 5, True, 'HS', 1],  # 7
                ['conv', c_d(160), c_d(960), 1, True, False, 'HS', 1],  # 7
                # 'pool', out_size
                ['pool', 1],  # 7
                ['conv', c_d(960), c_d(1280), 1, False, False, 'HS', 1],  # 1
                # 'dropout', p
                ['dropout', dropout],  # 1
                ['conv', c_d(1280), out_channels, 1, False, False, None, 1],  # 1
            )
        elif architecture == 'small':
            self.architecture = (
                # 'conv', in_ch, out_ch, k_size, batchnorm, sq_ext, non_linear, stride
                ['conv', in_channels, c_d(16), 3, True, False, 'HS', 2],  # 224
                # 'bneck', in_ch, exp_ch, out_ch, k_size, sq_exc, non_linear, stride
                ['bneck', c_d(16), c_d(16), c_d(16), 3, True, 'RE', 2],  # 112
                ['bneck', c_d(16), c_d(72), c_d(24), 3, False, 'RE', 2],  # 56
                ['bneck', c_d(24), c_d(88), c_d(24), 3, False, 'RE', 1],  # 28
                ['bneck', c_d(24), c_d(96), c_d(40), 5, True, 'HS', 2],  # 28
                ['bneck', c_d(40), c_d(240), c_d(40), 5, True, 'HS', 1],  # 14
                ['bneck', c_d(40), c_d(240), c_d(40), 5, True, 'HS', 1],  # 14
                ['bneck', c_d(40), c_d(120), c_d(48), 5, True, 'HS', 1],  # 14
                ['bneck', c_d(48), c_d(144), c_d(48), 5, True, 'HS', 1],  # 14
                ['bneck', c_d(48), c_d(288), c_d(96), 5, True, 'HS', 2],  # 14
                ['bneck', c_d(96), c_d(96), c_d(96), 5, True, 'HS', 1],  # 7
                ['bneck', c_d(96), c_d(576), c_d(96), 5, True, 'HS', 1],  # 7
                ['conv', c_d(96), c_d(576), 1, True, True, 'HS', 1],  # 7
                # 'pool', out_size
                ['pool', 1],  # 7
                ['conv', c_d(576), c_d(1024), 1, True, False, 'HS', 1],  # 1
                # 'dropout', p
                ['dropout', dropout],  # 1
                ['conv', c_d(1024), out_channels, 1, True, False, None, 1],  # 1
            )
        else:
            raise ValueError('size must be "large" or "small"')

    def weight_initialization(self):
        """
        Initialization model weight
        """
        for module in self.sequential.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def create_model(self,
                     classes_count: int,
                     in_channels: int = 3,
                     architecture: str = 'large',
                     alpha: float = 1.,
                     min_depth: int = 3,
                     dropout: float = 0.8
                     ):
        """
        Creates model with the specified parameters
        :param classes_count: (int or None) classes_count. ignored if architecture not is None
        :param in_channels: (int) input channels in model. Default: 3
        :param architecture: (str) Architecture should be "large" or "small"
        :param alpha: (float) coefficient of reduce depth size. Default: 1.
        :param min_depth: (int) max(min_depth, layer_depth * alpha). Default: 3
        :param dropout: (float) probability of an element to be zeroed. Default: 0.8
        """
        self.choice_model_architecture(architecture, in_channels, classes_count,
                                       alpha, min_depth, dropout)
        self._constructor()

    def _constructor(self):
        """
        Assembles the model
        """
        self.sequential = nn.Sequential()
        for ind, param in enumerate(self.architecture):
            layer_name = param[0]
            if layer_name == 'conv':
                self.sequential.add_module(f'{ind} {layer_name}', Convolution(*param[1:]))
            elif layer_name == 'bneck':
                self.sequential.add_module(f'{ind} {layer_name}', BottleNeck(*param[1:]))
            elif layer_name == 'pool':
                self.sequential.add_module(f'{ind} {layer_name}', nn.AdaptiveAvgPool2d(*param[1:]))
            elif layer_name == 'dropout':
                self.sequential.add_module(f'{ind} {layer_name}', nn.Dropout(*param[1:]))
        self.sequential.add_module('Flatten', nn.Flatten())
        self.weight_initialization()

    def save_model(self, file_name: str = 'model.pkl'):
        """
        save weights values and architecture in a binary file
        :param file_name: full name of a file. Default: "model.pkl"
        """
        save({'architecture': self.architecture,
              'state_dict': self.sequential.state_dict(),
              'labels': self.labels
              }, file_name)

    def load_model(self, file_name: str = 'model.pkl'):
        file = load(file_name)
        self.architecture = file['architecture']
        self._constructor()
        self.sequential.load_state_dict(file['state_dict'])
        self.labels = file['labels']

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)