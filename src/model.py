import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ChannelAttention(nn.Module):
    """Channel Attention Module for focusing on important feature channels"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention Module for focusing on important image regions"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class EnhancedConvBlock(nn.Module):
    """Enhanced Convolutional Block with Attention and Batch Normalization"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(EnhancedConvBlock, self).__init__()
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x

class ViolenceClassifier(nn.Module):
    """Enhanced Violence Classification Model with Attention Mechanisms and Transfer Learning"""
    def __init__(self, num_classes=4, use_pretrained=True):
        super(ViolenceClassifier, self).__init__()
        
        # Load pre-trained ResNet18 backbone
        weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
        resnet18 = models.resnet18(weights=weights)
        
        # Remove the last fully connected layer
        self.backbone = nn.Sequential(*list(resnet18.children())[:-1])
        
        # Feature extraction with enhanced blocks
        self.enhanced_conv1 = EnhancedConvBlock(3, 32, kernel_size=3)
        self.enhanced_conv2 = EnhancedConvBlock(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head with dropout and batch normalization
        self.fc_layers = nn.Sequential(
            nn.Linear(512 + 64, 256),  # 512 from ResNet18 + 64 from enhanced blocks
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Process through enhanced blocks
        enhanced = self.enhanced_conv1(x)
        enhanced = self.pool(enhanced)
        enhanced = self.enhanced_conv2(enhanced)
        enhanced = self.pool(enhanced)
        enhanced = self.global_avg_pool(enhanced)
        enhanced = enhanced.view(enhanced.size(0), -1)
        
        # Process through ResNet backbone
        backbone_features = self.backbone(x)
        backbone_features = backbone_features.view(backbone_features.size(0), -1)
        
        # Concatenate both feature streams
        combined_features = torch.cat([backbone_features, enhanced], dim=1)
        
        # Classification head
        output = self.fc_layers(combined_features)
        return output