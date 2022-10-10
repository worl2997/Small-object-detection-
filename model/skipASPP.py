import torch
import torch.nn.functional as F

from torch import nn
from collections import OrderedDict
from torch.nn import BatchNorm2d as bn

class skipASPP_(nn.Module):
    """
    * output_scale can only set as 8 or 16
    """
    def __init__(self, input_channels = 2048, dropout=0.1,mid_channels=128, out_channels=64):
        super(skipASPP_, self).__init__()

        dropout0 = dropout
        d_feature0 = mid_channels
        d_feature1 = out_channels
        #초기 입력feature->  backbone feature -> 2048
        # num1 -> 512 , num2 -> 128
        num_features = input_channels

        self.ASPP_3 = AsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=3, drop_out=dropout0, bn_start=False)

        self.ASPP_6 = AsppBlock(input_num=num_features + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=6, drop_out=dropout0, bn_start=True)

        self.ASPP_12 = AsppBlock(input_num=num_features + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=12, drop_out=dropout0, bn_start=True)

        self.ASPP_18 = AsppBlock(input_num=num_features + d_feature1 * 3, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=18, drop_out=dropout0, bn_start=True)

        self.ASPP_24 = AsppBlock(input_num=num_features + d_feature1 * 4, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=24, drop_out=dropout0, bn_start=True)

        self.make_output = nn.Conv2d(in_channels=num_features + d_feature1 * 4, out_channels=num_features, kernel_size=1)

        def forward(self, _input):
            # input -> C5
            aspp3 = self.ASPP_3(_input)
            feature = torch.cat((aspp3, _input), dim=1)

            aspp6 = self.ASPP_6(feature)
            feature = torch.cat((aspp6, feature), dim=1)

            aspp12 = self.ASPP_12(feature)
            feature = torch.cat((aspp12, feature), dim=1)

            aspp18 = self.ASPP_18(feature)
            feature = torch.cat((aspp18, feature), dim=1)

            aspp24 = self.ASPP_24(feature)

            out = torch.cat((aspp3,aspp6,aspp12,aspp18,aspp24), dim=1)
            out = self.make_output(out)
            return out


class AsppBlock(nn.Sequential):
    def __init__(self, input_num, num1, num2, dilation_rate, drop_out=0.1, bn_start=True):
        '''
        input_num -> 입력 채널 수
        num1 -> 중간 채널 수
        num2 -> 최종 output 채널 수
        '''
        super(AsppBlock, self).__init__()
        if bn_start:
            self.add_module('norm.1', bn(input_num, momentum=0.0003)),

        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm.2', bn(num1, momentum=0.0003)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),
        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(AsppBlock, self).forward(_input)
        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)
        return feature
