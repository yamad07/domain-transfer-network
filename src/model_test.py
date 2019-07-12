import torch
from models.model import Encoder
from models.model import Decoder
from models.model import Discriminator

if __name__ == "__main__":
    encoder = Encoder()
    img = torch.FloatTensor(1, 1, 32, 32)
    output = encoder(img)
    print(output.size())

    decoder = Decoder()
    img = decoder(output)
    print(img.size())

    discriminator = Discriminator()
    output = discriminator(img)
    print(output.size())
