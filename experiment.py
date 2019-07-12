from src.models.model import Encoder, Decoder, Discriminator
from src.trainer import Trainer
from src.dataset import MNISTToSVHN
from torch.utils.data import DataLoader

import os

mnist_to_svhn_dataset = MNISTToSVHN(
    mnist_img_dir=os.path.abspath("./mnist"),
    svhn_img_dir=os.path.abspath("./svhn"),
)
data_loader = DataLoader(mnist_to_svhn_dataset, batch_size=10, shuffle=True)
trainer = Trainer(
    encoder=Encoder(),
    decoder=Decoder(),
    discriminator=Discriminator(),
    data_loader=data_loader,
    a=100,
    b=1,
    c=0.05
)
trainer.train(100)
