import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False), nn.BatchNorm2d(out_c)
        self.conv2, self.bn2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False), nn.BatchNorm2d(out_c)
        self.skip = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c)) if stride!=1 or in_c!=out_c else nn.Sequential()
        
    def forward(self, x): 
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.skip(x))

class Model_depth(nn.Module):
    def __init__(self, dim_in, dim_out, is_mini=False):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        ch = [16, 32, 64, 128, 256] if is_mini else [32, 64, 128, 256, 512]
        self.enc1, self.enc2, self.enc3, self.enc4 = ResBlock(dim_in, ch[0]), ResBlock(ch[0], ch[1]), ResBlock(ch[1], ch[2]), ResBlock(ch[2], ch[3])
        self.bottleneck = ResBlock(ch[3], ch[4])
        self.up1, self.dec1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ch[4], ch[3], 1)), ResBlock(ch[4], ch[3])
        self.up2, self.dec2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ch[3], ch[2], 1)), ResBlock(ch[3], ch[2])
        self.up3, self.dec3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ch[2], ch[1], 1)), ResBlock(ch[2], ch[1])
        self.up4, self.dec4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ch[1], ch[0], 1)), ResBlock(ch[1], ch[0])
        self.final = nn.Sequential(nn.Conv2d(ch[0], dim_out, 1), nn.Softplus())
        
    def forward(self, x, return_features=True):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))
        d1 = self.dec1(torch.cat([self.up1(b), s4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), s3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), s2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), s1], dim=1))
        return (self.final(d4), [s1, s2, s3, s4, b]) if return_features else self.final(d4)

class Timm_Depth(nn.Module):
    def __init__(self, dim_out=1, backbone='resnet18'):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=True, features_only=True)
        ch = self.encoder.feature_info.channels()
        self.up1, self.dec1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ch[4], ch[3], 1)), ResBlock(ch[3]*2, ch[3])
        self.up2, self.dec2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ch[3], ch[2], 1)), ResBlock(ch[2]*2, ch[2])
        self.up3, self.dec3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ch[2], ch[1], 1)), ResBlock(ch[1]*2, ch[1])
        self.up4, self.dec4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ch[1], ch[0], 1)), ResBlock(ch[0]*2, ch[0])
        self.final = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.Conv2d(ch[0], dim_out, 1), nn.Softplus())
        
    def forward(self, x, return_features=True):
        feat = self.encoder(x)
        d1 = self.dec1(torch.cat([self.up1(feat[4]), feat[3]], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), feat[2]], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), feat[1]], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), feat[0]], dim=1))
        return (self.final(d4), feat) if return_features else self.final(d4)