import torch
import torch.nn as nn
import torchvision.models as models
import segmentation_models_pytorch as smp

class XRayNet(nn.Module):
    def __init__(self):
        super(XRayNet, self).__init__()

        # U-Net using segmentation_models_pytorch
        self.seg = nn.Module()
        self.seg.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

        # DenseNet121 for classification with dropout
        densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_features = densenet.classifier.in_features
        densenet.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 1)
        )

        self.cls = densenet

    def forward(self, x):
        # Get segmentation mask
        mask = self.seg.unet(x)
        
        # Apply mask
        masked = x * mask
        
        # Classification
        cls_output = self.cls(masked)
        
        return mask, cls_output