import torch
import torchvision
from torch import nn


def build_model(args, pretrained=True):
    if args.backbone == "resnet50":
        net = ResNet50Mod(args.n_bits, args.n_classes, pretrained).to(args.device)
        return net, 0

    raise NotImplementedError(f"not support: {args.backbone}")


class ResNet50Mod(nn.Module):
    def __init__(self, n_bits, n_classes, pretrained=True):
        super().__init__()

        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.cnn = torchvision.models.resnet50(weights=weights)
        self.dim_feature = self.cnn.fc.in_features
        self.dropout = nn.Dropout(p=0.1)

        # for hashing
        self.fc = nn.Linear(self.dim_feature, n_bits)

        # classification
        self.classifier = nn.Linear(n_bits, n_classes)

    def forward(self, x: torch.Tensor):
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = self.cnn.layer4(x)

        x = self.cnn.avgpool(x)
        x = torch.flatten(x, 1)

        embeddings = self.dropout(self.fc(x))
        logits = self.classifier(embeddings)

        return embeddings, logits


if __name__ == "__main__":
    x = torch.randn(64, 3, 224, 224)
    model = ResNet50Mod(16, 10)
    out = model(x)
    for x in out:
        print(x.shape)
