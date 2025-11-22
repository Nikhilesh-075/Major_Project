import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

class EfficientNetB4(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        if pretrained:
            self.base_model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        else:
            self.base_model = efficientnet_b4(weights=None)

        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
