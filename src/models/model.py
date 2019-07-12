import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.cnn import encoder_layer
from .layers.cnn import decoder_layer
from .layers.cnn import discriminator_layer


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.c1 = encoder_layer(3, 64, 3)
        self.c2 = encoder_layer(64, 128, 3)
        self.c3 = encoder_layer(128, 256, 3)
        self.c4 = nn.Sequential(
            nn.Conv2d(256,
                      128,
                      stride=2,
                      kernel_size=4,
                      padding=0,
                      ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size = x.size(0)
        h = self.c1(x)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        h = h.view(batch_size, -1)
        return h


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_block1 = decoder_layer(128, 512, 4, 0)
        self.conv_block2 = decoder_layer(512, 256, 4, 1)
        self.conv_block3 = decoder_layer(256, 128, 4, 1)
        self.conv4 = nn.ConvTranspose2d(128, 3, kernel_size=4,
                                        stride=2, padding=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 128, 1, 1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv4(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_block1 = discriminator_layer(3, 128)
        self.conv_block2 = discriminator_layer(128, 256)
        self.conv_block3 = discriminator_layer(256, 512)
        self.c4 = nn.Sequential(
            nn.Conv2d(512,
                      3,
                      stride=2,
                      kernel_size=4,
                      padding=0,
                      ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size = x.size(0)
        h = self.conv_block1(x)
        h = self.conv_block2(h)
        h = self.conv_block3(h)
        h = self.c4(h)
        return F.log_softmax(h.view(batch_size, -1), dim=1)
