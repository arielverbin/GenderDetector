import torch
import torch.nn as nn
import torchvision.transforms as t


# define a block of residual network
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.0):
        super(Block, self).__init__()

        # down-sample layer for the residual connection, if stride > 1
        if stride > 1:
            self.downsampling = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsampling = None

        # two convolution layers with batch normalization and ReLU activation
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    # forward pass of the block
    def forward(self, x):
        # Save the input as the identity tensor for the residual connection
        identity = x

        # Perform the first convolutional layer operation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Perform the second convolutional layer operation
        x = self.conv2(x)
        x = self.bn2(x)

        # If down-sample layer is defined, apply it to the identity
        # tensor for the residual connection layer
        if self.downsampling:
            identity = self.downsampling(identity)

        # Add the identity tensor to the output of the second convolutional layer
        x += identity

        # Apply dropout after the residual connection
        x = self.dropout(x)

        # Apply the ReLU activation function to the sum of the convolutional layers and the identity tensor
        x = self.relu(x)

        # Return the output of the ResidualBlock
        return x


# define a ResNet layer
class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout_rate):
        super(ResNetLayer, self).__init__()
        layers = [
                Block(in_channels, out_channels, stride, dropout_prob=dropout_rate),
                Block(out_channels, out_channels, stride, dropout_prob=dropout_rate)
        ]
        # create a sequential container of the two blocks
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # pass input through the two blocks of the sequential container
        return self.layers(x)


# define the ResNet18 architecture
class ResNet18(nn.Module):
    def __init__(self, dropout_rate):
        super(ResNet18, self).__init__()
        # data augmentation layer
        self.augment = torch.nn.Sequential(
            t.RandomPerspective(distortion_scale=0.2, p=0.7),
            t.RandomHorizontalFlip(p=0.3),
            t.RandomRotation(degrees=(0, 180)),
            t.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        # initial convolution layer, batch normalization, ReLU activation and max pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channels = 64
        # four ResNet layers with different number of blocks
        self.layer1 = ResNetLayer(64, 64, stride=1, dropout_rate=dropout_rate)
        self.layer2 = ResNetLayer(64, 128, stride=2, dropout_rate=dropout_rate)
        self.layer3 = ResNetLayer(128, 256, stride=2, dropout_rate=dropout_rate)
        self.layer4 = ResNetLayer(256, 512, stride=2, dropout_rate=dropout_rate)
        # global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1)
        # Sigmoid function for normalizing output to probabilities
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply data augmentation to the input
        x = self.augment(x)
        # Apply the first convolution layer
        x = self.conv1(x)
        # Apply batch normalization to the output of the first convolution layer
        x = self.bn1(x)
        # Apply the ReLU activation function
        x = self.relu(x)
        # Apply max pooling to the output of the ReLU activation function
        x = self.maxpool(x)

        # Apply the four ResNet layers in sequence
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Apply adaptive average pooling to the output of the final ResNet layer
        x = self.avgpool(x)
        # Flatten the output of the adaptive average pooling layer
        x = x.reshape(x.shape[0], -1)
        # Apply the fully connected layer with a single output unit
        x = self.fc1(x)

        # Apply the sigmoid activation function to the output of the fully connected layer
        return self.sigmoid(x)
