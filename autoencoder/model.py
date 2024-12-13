import torch
import torch.nn as nn

class ResNetAutoencoder(nn.Module):
    def __init__(self, in_channels, use_dropout=False, layers=[2, 2, 2, 2],
                 factor_fewer_resnet_channels=8, activation_fct="relu", groups=1, width_per_group=64,
                 replace_stride_with_dilation=None):
        super(ResNetAutoencoder, self).__init__()

        block = BasicBlock
        up_block = UpBasicBlock
        self.activation_fct = activation_fct

        self.inplanes = int(64 / factor_fewer_resnet_channels)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group

        if use_dropout:
            self.dropout_values = nn.Dropout(p=0.2, inplace=False)
            self.dropout_channels = nn.Dropout2d(p=0.2, inplace=False)
        else:
            self.dropout_values = nn.Identity()
            self.dropout_channels = nn.Identity()

        # Encoder
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.inplanes,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.activation = nn.ReLU(inplace=True) if activation_fct == "relu" else nn.Tanh()
        self.maxpool = nn.Identity()  # Changed from MaxPool2d to Identity

        self.layer1 = self._make_layer(block, int(64 / factor_fewer_resnet_channels), layers[0])
        self.layer2 = self._make_layer(block, int(128 / factor_fewer_resnet_channels), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(256 / factor_fewer_resnet_channels), layers[2], stride=2)
        self.layer4 = self._make_layer(block, int(512 / factor_fewer_resnet_channels), layers[3], stride=2)

        # Decoder
        self.up_layer4 = self._make_up_layer(up_block, int(512 / factor_fewer_resnet_channels), int(256 / factor_fewer_resnet_channels), layers[3], stride=2)
        self.up_layer3 = self._make_up_layer(up_block, int(256 / factor_fewer_resnet_channels), int(128 / factor_fewer_resnet_channels), layers[2], stride=2)
        self.up_layer2 = self._make_up_layer(up_block, int(128 / factor_fewer_resnet_channels), int(64 / factor_fewer_resnet_channels), layers[1], stride=2)
        self.up_layer1 = self._make_up_layer(up_block, int(64 / factor_fewer_resnet_channels), int(64 / factor_fewer_resnet_channels), layers[0], stride=2)

        self.upconv = nn.ConvTranspose2d(
            in_channels=int(64 / factor_fewer_resnet_channels),
            out_channels=in_channels,
            kernel_size=3,
            stride=1,          # Changed stride from 2 to 1
            padding=1,
            output_padding=0,  # Changed output_padding from 1 to 0
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if self.activation_fct == 'relu':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif self.activation_fct == 'tanh':
                    nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, self.dilation, self.activation_fct))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, activation_fct=self.activation_fct))

        return nn.Sequential(*layers)

    def _make_up_layer(self, block, inplanes, planes, blocks, stride=1):
        upsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv1x1(inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, upsample, self.activation_fct))
        inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, activation_fct=self.activation_fct))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x = self.dropout_values(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)


        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x3 = self.dropout_channels(x3)
        x4 = self.layer4(x3)

        # Decoder
        x = self.up_layer4(x4)
        x = self.up_layer3(x)
        x = self.up_layer2(x)
        x = self.up_layer1(x)
        x = self.upconv(x)
        x = self.sigmoid(x)
        return x
    
    def encode(self, x):
        # Encoder
        x = self.dropout_values(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x3 = self.dropout_channels(x3)
        x4 = self.layer4(x3)
        return x4

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def deconv1x1(in_planes, out_planes, stride=1):
    """1x1 deconvolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride,
                              output_padding=stride-1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, activation_fct="relu"):
        super(BasicBlock, self).__init__()
        self.activation = nn.ReLU(inplace=True) if activation_fct == "relu" else nn.Tanh()
        self.conv1 = conv3x3(inplanes, planes, stride, groups, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out

class UpBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, activation_fct="relu"):
        super(UpBasicBlock, self).__init__()
        self.activation = nn.ReLU(inplace=True) if activation_fct == "relu" else nn.Tanh()
        self.conv_transpose1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, stride=stride,
                                                  padding=1, output_padding=stride-1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv_transpose2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=1,
                                                  padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv_transpose1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv_transpose2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.activation(out)

        return out