"""
A Convolutional Variational Autoencoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=64 * 16 * 16, zDim=256):
        super(VAE, self).__init__()

        # Initializing the 3 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 16, 5)
        self.encConv2 = nn.Conv2d(16, 32, 5)
        self.encConv3 = nn.Conv2d(32, 64, 5)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Initializing the fully-connected layer and 3 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(64, 32, 5)
        self.decConv2 = nn.ConvTranspose2d(32, 16, 5)
        self.decConv3 = nn.ConvTranspose2d(16, imgChannels, 5)

    def encoder(self, x):
        # Input is fed into 3 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        # print(f'Size of encConv1: {x.shape}')
        x = F.relu(self.encConv2(x))
        # print(f'Size of encConv2: {x.shape}')
        x = F.relu(self.encConv3(x))
        # print(f'Size of encConv3: {x.shape}')
        x = x.view(-1, 64 * 16 * 16)
        mu = self.encFC1(x)
        # print(f'Size of mu (encFC1): {mu.shape}')
        logVar = self.encFC2(x)
        # print(f'Size of logVar (encFC2): {logVar.shape}')
        return mu, logVar

    def reparameterize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        # z is fed back into a fully-connected layers and then into three transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 64, 16, 16)
        x = F.relu(self.decConv1(x))
        x = F.relu(self.decConv2(x))
        x = torch.sigmoid(self.decConv3(x))
        # print(f'Output of the decoder: {x}')
        return x

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        # print(f'Input of the decoder: {z}')
        out = self.decoder(z)
        return out, mu, logVar
