import torchvision.models as models
import torch.nn as nn

def EfficientNet_B7(num_classes=2, in_channel=3, pretrained=True):
    weights = models.EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else False
    model = models.efficientnet_b7(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model