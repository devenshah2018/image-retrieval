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

        # Freeze early layers to prevent overfitting
        # Only train the last few layers (fine-tuning)
        for param in list(self.resnet.parameters())[:-8]:  # Freeze all except the last 2 blocks
            param.requires_grad = False

        # Replace the final fully connected layer
        # ResNet18's final layer input features is 512
        self.resnet.fc = nn.Identity()  # Remove the final FC layer

        # Add our custom classifier with dropout
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True)
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