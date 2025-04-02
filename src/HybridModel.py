import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import segmentation_models_pytorch as smp

# Automatically use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNetSegmentation(nn.Module):
    def __init__(self):
        super(UNetSegmentation, self).__init__()
        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )

    def forward(self, x):
        return torch.sigmoid(self.unet(x))

class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.seg = UNetSegmentation().to(device)
        self.classifier = mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.classifier.classifier[3] = nn.Linear(
            self.classifier.classifier[3].in_features, 1
        )

    def forward(self, x):
        mask = self.seg(x)
        masked = x * (mask > 0.5).float()  # Optional: binarize the mask
        return mask, self.classifier(masked)
