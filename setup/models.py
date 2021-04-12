import torch
import torch.nn as nn
import torch.nn.functional as F

# ResNet-18 backbone: This is paper's version. Also used here: https://github.com/microsoft/snca.pytorch/blob/master/models/resnet_cifar.py
# dfrnt from torchvision model, CHECK
"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18a(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

#---------------------------------------------------------------------------------

class SimclrContrastiveModel(nn.Module):
    def __init__(self, backbone, head='MLP', featuresDim=128, backboneDim=512):
        super(SimclrContrastiveModel, self).__init__()
        self.backbone = backbone
        self.backboneDim = backboneDim
        self.head = head  # need? if linear not used, remove
        # simCLR uses 2 layer MLP head --- check paper
        # nn.Linear(input sample size, output sample size)
        # self.contrastiveHead = nn.Linear(self.backboneDim, featuresDim) # just for testing
        self.contrastiveHead = nn.Sequential(nn.Linear(self.backboneDim, self.backboneDim),
                                             nn.ReLU(), nn.Linear(self.backboneDim, featuresDim))

    def forward(self, x):
        features = self.contrastiveHead(self.backbone(x))
        features = F.normalize(features, dim=1)
        return features

#----------------------------------------------------------------------

class ClusteringModel(nn.Module):
  def __init__(self, backbone, numClasses, numHeads=1, backboneDim=512):
    super(ClusteringModel, self).__init__()
    self.backbone = backbone
    self.backboneDim = backboneDim
    self.numHeads = numHeads
    self.cluster_head = nn.ModuleList([nn.Linear(self.backboneDim, numClasses) for _ in range(self.numHeads)])

  def forward(self, x):
      features = self.backbone(x)
      out = [cluster_head(features) for cluster_head in self.cluster_head]

      # add?
      return out

#------------------------------------------------------------------------------
def get_model(step, pretrained_weights=None, numClasses=None):
    # Get backbone
    # import resnet18 backbone from torchvision:
    # resnet18 = torchvision.models.resnet18(pretrained=False)
    # resnet18_ft = nn.Sequential(*(list(resnet18.children())[0:9])) # remove last layer and retain feature extractor
    backbone = resnet18a()
    # backbone = resnet18_ft

    if step == "simclr":
        # If pretext task, get simclr contrastive model
        model = SimclrContrastiveModel(backbone)
    # If scan or selflabel task, get clustering model
    if step == "scan" or step == "selflabel":
        model = ClusteringModel(backbone, numClasses)  # removed numHeads

    # Check for pretrained weights
    if pretrained_weights is not None:
        state = torch.load(pretrained_weights, map_location='cpu')
        # In SCAN step, weights are transferred from pretext step
        if step == 'scan':
            weights = model_load_state_dict(state, strict=False)
            # if strict=False, previous model and new model in which weights will be used don't have to be identical
        # In selflabel step, weight are transferred from SCAN step         # NEW
        if step == 'selflabel':
            # CHECK: continue with best head and pop others, but only using one head??
            weights = model_load_state_dict(state, strict=True)

    return model


