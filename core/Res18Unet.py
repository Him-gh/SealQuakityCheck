import torch
from torch import nn
import torchvision.models as models



class ResNet18Segmentation(nn.Module):
    def __init__(self, num_classes=2,pretrained=False):
        super(ResNet18Segmentation, self).__init__()
        
        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            resnet = models.resnet18(weights=None)
        
        # 编码器各层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 输出尺寸: H/4, W/4
        self.layer2 = resnet.layer2  # 输出尺寸: H/8, W/8
        self.layer3 = resnet.layer3  # 输出尺寸: H/16, W/16
        self.layer4 = resnet.layer4  # 输出尺寸: H/32, W/32
        
        # 解码器（带跳跃连接）
        self.decoder5 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.decoder4 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64 + 64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, x):
        input_size = x.shape[-2:]
        
        # Stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        e1 = x  # [B, 64, H/2, W/2]
        x = self.maxpool(x)
        e2 = self.layer1(x)  # [B, 64, H/4, W/4]
        e3 = self.layer2(e2)  # [B, 128, H/8, W/8]
        e4 = self.layer3(e3)  # [B, 256, H/16, W/16]
        e5 = self.layer4(e4)  # [B, 512, H/32, W/32]
        
        # 解码器（带跳跃连接和尺寸对齐）
        d5 = self.decoder5(e5)
        if d5.shape[-2:] != e4.shape[-2:]:
            d5 = nn.functional.interpolate(d5, size=e4.shape[-2:], mode='bilinear', align_corners=True)
        d5 = torch.cat([d5, e4], dim=1)

        d4 = self.decoder4(d5)
        if d4.shape[-2:] != e3.shape[-2:]:
            d4 = nn.functional.interpolate(d4, size=e3.shape[-2:],mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e3], dim=1)
    
        d3 = self.decoder3(d4)
        if d3.shape[-2:] != e2.shape[-2:]:
            d3 = nn.functional.interpolate(d3, size=e2.shape[-2:],mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e2], dim=1)
        
        d2 = self.decoder2(d3)
        if d2.shape[-2:] != e1.shape[-2:]:
            d2 = nn.functional.interpolate(d2, size=e1.shape[-2:],mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e1], dim=1)
    
        d1 = self.decoder1(d2)
        if d1.shape[-2:] != input_size:
            d1 = nn.functional.interpolate(d1, size=input_size,mode='bilinear', align_corners=True)
        
        output = self.classifier(d1)
        return output

