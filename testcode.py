from torchvision import models
import torch.nn as nn

resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 7)
print(resnet18)