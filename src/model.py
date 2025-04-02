import torch
import torch.nn as nn
import torchvision.models as models

class XRayNet(nn.Module):
    def __init__(self, num_classes=2):
        super(XRayNet, self).__init__()
        # Load pretrained DenseNet
        self.densenet = models.densenet121(pretrained=True)
        
        # Modify the classifier for our needs
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Add segmentation head
        self.seg = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
    def forward(self, x):
        # Get features from DenseNet
        features = self.densenet.features(x)
        
        # Classification branch
        out = torch.flatten(features, 1)
        out = self.classifier(out)
        
        # Segmentation branch
        seg_out = self.seg(features)
        
        return out, seg_out

def load_model(model_path):
    try:
        model = XRayNet()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None 