import torch.nn as nn
import torch.nn.functional as F
import torch 

class BasicBlock(nn.Module):
    def __init__(self,in_features,out_features,stride=1,mid_kernel_size=3,final_bn=True,final_relu=True):
        super().__init__()
        self._stride = stride
        self._final_bn = final_bn
        self._final_relu = final_relu
        self._same_feature_depth = (in_features == out_features)
        if stride != 1:
            self.conv0 = nn.Conv2d(in_features,in_features,kernel_size=3,stride=stride,groups=in_features,bias=False)
            self.bn0 = nn.BatchNorm2d(in_features)
        self.conv1 = nn.Conv2d(in_features,out_features*3,kernel_size=1)
        #self.bn1 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features*3,out_features*3,kernel_size=mid_kernel_size,groups=out_features*3,padding=mid_kernel_size//2,bias=False)
        self.bn2 = nn.BatchNorm2d(out_features*3)
        self.conv3 = nn.Conv2d(out_features*3,out_features,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_features)
    def forward(self,x):
        y = x
        if self._stride != 1:
            y = F.relu(self.bn0(self.conv0(x)))

        y = self.conv1(y)
        y = self.conv2(y)
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.conv3(y)
        if self._final_bn:
            y = self.bn3(y)
        if self._final_relu:
            y = F.relu(y)
        if self._stride == 1 and self._final_relu and self._same_feature_depth:
            y = y + x 
        return y


class SmallNet(nn.Module):
    def __init__(self,num_blocks=3,num_layers_per_block=4,first_feature_size=40):
        super().__init__()
        layers = [nn.Conv2d(3,first_feature_size,kernel_size=3,padding=1,bias=False)]
        layers += [nn.BatchNorm2d(first_feature_size)]
        layers += [nn.ReLU(inplace=True)]
        feature_size = first_feature_size
        for block in range(num_blocks):
            layers += [BasicBlock(feature_size,2*feature_size,stride=2)]
            feature_size = 2*feature_size
            for layer in range(num_layers_per_block-1):
                if block %2 ==1 and layer %2 ==0:
                    layers += [BasicBlock(feature_size,feature_size,mid_kernel_size=5)]
                else:
                    layers += [BasicBlock(feature_size,feature_size)]
        layers += [BasicBlock(feature_size,200)] #,final_bn=False,final_relu=False)]
        layers += [nn.AdaptiveAvgPool2d(1),nn.Flatten()]
        layers += [nn.Linear(200,10)]

        self.features = nn.Sequential(*layers)
        self.layers = layers

    def forward(self,x):
        return torch.squeeze(self.features(x))



