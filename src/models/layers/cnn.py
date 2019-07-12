import torch.nn as nn


def encoder_layer(input_dim, output_dim, kernel_size, padding=1):
    return nn.Sequential(
        nn.Conv2d(input_dim,
                  output_dim,
                  stride=2,
                  kernel_size=kernel_size,
                  padding=padding,
                  ),
        nn.BatchNorm2d(output_dim),
        nn.ReLU(inplace=True),
    )


def decoder_layer(input_dim, output_dim,
                  kernel_size, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(input_dim, output_dim,
                           kernel_size=kernel_size, stride=2, padding=padding),
        nn.BatchNorm2d(output_dim),
        nn.ReLU(inplace=True),
    )


def discriminator_layer(input_dim, output_dim):
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=3,
                  stride=2, padding=1),
        nn.BatchNorm2d(output_dim),
        nn.LeakyReLU(),
    )
