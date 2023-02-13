import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    This model maps an image to an embedding vector. This embedding vector can be used as the first
    input that is shown to the decoder of a Seq2Seq model
    """
    def __init__(self, embedding_dim, model=models.resnet50, model_input_dim=224, patch_size=224, pretrained=True, requires_grad=False, device='cpu', **kwargs):
        """
        :param embedding_dim: Dimension of the embedding space (outputs of this module)
        :param model: The model that shouldbe used for encoding the image
        :param pretrained: True if pretrained weights should be used for the model
        :param required_grad: True if the CNN should be trained
        """
        super(EncoderCNN, self).__init__()
        # Upsampling layer
        self.scale_factor = model_input_dim // patch_size
        if self.scale_factor > 1:
            upsample = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
        # CNN model from torchvision
        model = model(pretrained=pretrained)
        # Training the CNN parameters is optional
        for param in model.parameters():
            param.requires_grad_(requires_grad)
        
        # Remove last layer from CNN
        modules = list(model.children())[:-1]
        # Run image through the CNN
        self.cnn = nn.Sequential(*modules)
        # Map outputs of CNN to the embedding space
        self.embed = nn.Linear(model.fc.in_features, embedding_dim)

    def forward(self, images):
        if self.scale_factor > 1:
            images = self.upsample(images)
        features = self.cnn(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
    
class TinyEncoderCNN(nn.Module):
    def __init__(self, patch_size, embedding_dim, model_input_dim=224, dropout=0.1, activation=F.relu, device='cpu', **kwargs):
        """
        :param embedding_dim: Dimension of the embedding space (outputs of this module)
        :param model: The model that shouldbe used for encoding the image
        :param pretrained: True if pretrained weights should be used for the model
        :param required_grad: True if the CNN should be trained
        """
        super(TinyEncoderCNN, self).__init__()
        
        # Hyperparameters
        self.model_input_dim=model_input_dim
        self.embedding_dim=embedding_dim
        self.patch_size=patch_size
        self.activation = activation
        
        # Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, stride=2, bias=False)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=2, stride=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=2, stride=2, bias=False)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=2, stride=2, bias=False)
        self.softconv = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc1 = nn.Linear((model_input_dim // 16)**2, (model_input_dim // 16)**2)
        self.fc2 = nn.Linear((model_input_dim // 16)**2, (model_input_dim // 16)**2)
        self.fc3 = nn.Linear((model_input_dim // 16)**2, embedding_dim)
        
    def forward(self, x):
        # Run image through the convolutional layers
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        # Perform 1x1 convolution for dimensionality reduction
        x = self.activation(self.softconv(x))
        # Flatten all dimensions except batch
        x = torch.flatten(x, 1)
        # Embed the outputs of the convolutional layers
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x