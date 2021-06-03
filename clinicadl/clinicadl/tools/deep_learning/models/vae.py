import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from .vae_structure import VAE_Encoder, VAE_Decoder

class VanillaVAE(nn.Module):
    def __init__(self, input_shape, feature_size,
                 latent_size, latent_dim,
                 n_conv, io_layer_channel):
        super(VanillaVAE, self).__init__()
        
        self.input_shape = input_shape
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.latent_dim = latent_dim
        self.n_conv = n_conv
        self.io_layer_channel = io_layer_channel

        self.encoder = VAE_Encoder(
            input_shape=self.input_shape,
            feature_size=self.feature_size,
            latent_dim=self.latent_dim,
            n_conv=self.n_conv,
            first_layer_channels=self.io_layer_channel
        )

        if self.latent_dim==1:
            # hidden => mu
            self.fc1 = nn.Linear(self.feature_size, self.latent_size)
            # hidden => logvar
            self.fc2 = nn.Linear(self.feature_size, self.latent_size)
        elif self.latent_dim==2:
            # hidden => mu
            self.fc1 = nn.Conv2d(self.feature_size, 1,
                                 4, stride=1, padding=0, bias=False)
            # hidden => logvar
            self.fc2 = nn.Conv2d(self.feature_size, 1,
                                 4, stride=1, padding=0, bias=False)
        else:
            raise AttributeError("Bad latent dimension specified. Latent dimension must be 1 or 2")

        self.decoder = VAE_Decoder(
            input_shape=self.input_shape,
            latent_size=self.latent_size,
            feature_size=self.feature_size,
            latent_dim=self.latent_dim,
            n_conv=self.n_conv,
            last_layer_channels=self.io_layer_channel
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def reparameterize_eval(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
