from torchvision.models import inception_v3
from torch import nn

model = inception_v3(weights = "IMAGENET1K_V1", num_classes = 3)