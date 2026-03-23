import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

class ResNetTransferModel(nn.Module):
    def __init__(self, num_classes=101, embedding_size=128, pretrained=True):
        super(ResNetTransferModel, self).__init__()

        # Load pre-trained ResNet18 (smaller than ResNet50 for faster training)
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = resnet18(weights=weights)

        # Match transfer-learning notebook: freeze backbone, then unfreeze layer3, layer4, avgpool
        for param in self.resnet.parameters():
            param.requires_grad = False
        for module in (self.resnet.layer3, self.resnet.layer4, self.resnet.avgpool):
            for param in module.parameters():
                param.requires_grad = True

        # Replace the final fully connected layer
        # ResNet18's final layer input features is 512
        self.resnet.fc = nn.Identity()  # Remove the final FC layer

        # Embedding head: BN + ReLU + dropout (notebook) before classifier
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )

        # Classification layer
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        # Extract features from ResNet
        features = self.resnet(x)
        # Get embedding
        embedding = self.embedding(features)
        # Get class predictions
        logits = self.classifier(embedding)
        return logits

    def extract_features(self, x):
        """Extract feature embeddings for image retrieval"""
        features = self.resnet(x)
        embedding = self.embedding(features)
        # Normalize embedding to unit length for better similarity search
        normalized_embedding = F.normalize(embedding, p=2, dim=1)
        return normalized_embedding