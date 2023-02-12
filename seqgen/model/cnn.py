import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    This model maps an image to an embedding vector. This embedding vector can be used as the first
    input that is shown to the decoder of a Seq2Seq model
    """
    def __init__(self, embedding_dim, model=models.resnet50, pretrained=True, requires_grad=False, device='cpu', **kwargs):
        """
        :param embedding_dim: Dimension of the embedding space (outputs of this module)
        :param model: The model that shouldbe used for encoding the image
        :param pretrained: True if pretrained weights should be used for the model
        :param required_grad: True if the CNN should be trained
        """
        super(EncoderCNN, self).__init__()
        model = model(pretrained=pretrained)
        # We don't want to train the 
        for param in model.parameters():
            param.requires_grad_(requires_grad)
        
        modules = list(model.children())[:-1]
        # Run image through the CNN
        self.cnn = nn.Sequential(*modules)
        # Map outputs of CNN to the embedding space
        self.embed = nn.Linear(model.fc.in_features, embedding_dim)

    def forward(self, images):
        features = self.cnn(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features