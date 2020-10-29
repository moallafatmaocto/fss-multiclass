import torchvision.models as models
from torch import nn


class CNNEncoder(nn.Module):
    def __init__(self, class_num):
        super(CNNEncoder, self).__init__()
        features = list(models.vgg16_bn(pretrained=False).features)
        self.layer1 = nn.Sequential(
            nn.Conv2d(class_num + 3, 64, kernel_size=3, padding=1)
        )
        self.features = nn.ModuleList(features)[1:]

    def forward(self, x):
        feature_list = []
        x = self.layer1(x)

        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {4, 11, 21, 31, 41}:  # Batch Norm 2D layers respectively : 64, 128, 256, 512, 512
                feature_list.append(x)

        return x, feature_list
