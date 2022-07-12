import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, resBlock=ResBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(resBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(resBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(resBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(resBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        rep = out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        prob = F.softmax(self.fc(out), dim=1)
        return rep.view(out.size(0), -1), prob


class Siamese(nn.Module):
    def __init__(self, net):
        super(Siamese, self).__init__()
        self.net = net
        self.net.fc.requires_grad_(False)
        self.fc = nn.Linear(4 * 4 * 512, 1)

    def forward(self, input1, input2):
        output1, _ = self.net(input1)
        output2, _ = self.net(input2)
        l1 = torch.abs(output1 - output2)
        l1 = self.fc(l1)
        return output1, output2, l1.view(-1)


class ContrastLoss(nn.Module):
    def __init__(self, margin=3.0):
        super(ContrastLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        d = F.pairwise_distance(output1, output2, keepdim=True)
        loss = torch.mean((1 - label) * torch.pow(d, 2) + label * torch.pow(torch.clamp(self.margin - d, min=0.0), 2))
        return loss
