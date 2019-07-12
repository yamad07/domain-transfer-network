import torch.optim as optim
import torch
import torch.nn
import torch.nn.functional as F
import torchvision


class Trainer:
    def __init__(self, encoder, decoder, discriminator, data_loader, a, b, c):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.data_loader = data_loader
        self.a = a
        self.b = b
        self.c = c

        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=0.01)
        self.decoder_optim = optim.Adam(self.decoder.parameters(), lr=0.01)
        self.discriminator_optim = optim.Adam(
            self.discriminator.parameters(), lr=0.01)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epoch):

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.discriminator.to(self.device)
        for e in range(epoch):
            for svhn_img, svhn_labels, mnist_img, mnist_labels in self.data_loader:
                svhn_img = svhn_img.to(self.device)
                mnist_img = mnist_img.to(self.device)
                # mnist_img ... [batch_size, 3, img_size, img_size]
                batch_size = mnist_img.size(0)

                # Optimize Discriminator
                encoded_mnist_img = self.encoder(mnist_img)
                decoded_mnist_img = self.decoder(encoded_mnist_img)
                prob_1 = self.discriminator(decoded_mnist_img)
                labels_1 = torch.zeros(batch_size).long().to(self.device)
                # prob_1 .. . [batch_size, 3]
                # labels_1 .. . [batch_size]
                discriminator_1_loss = F.nll_loss(prob_1, labels_1)

                encoded_svhn_img = self.encoder(svhn_img)
                decoded_svhn_img = self.decoder(encoded_svhn_img)
                prob_2 = self.discriminator(decoded_svhn_img)
                labels_2 = torch.ones(batch_size).long().to(self.device)

                discriminator_2_loss = F.nll_loss(prob_2, labels_2)

                prob_3 = self.discriminator(svhn_img)
                labels_3 = (labels_2.float() * 2).long().to(self.device)

                discriminator_3_loss = F.nll_loss(prob_3, labels_3)

                discriminator_loss = discriminator_1_loss + \
                    discriminator_2_loss + discriminator_3_loss
                discriminator_loss.backward()
                self.discriminator_optim.step()

                # gang_loss
                encoded_mnist_img = self.encoder(mnist_img)
                decoded_mnist_img = self.decoder(encoded_mnist_img)
                prob_gang_s = self.discriminator(decoded_mnist_img)

                gang_s_loss = F.nll_loss(prob_gang_s, labels_3)

                encoded_svhn_img = self.encoder(svhn_img)
                decoded_svhn_img = self.decoder(encoded_svhn_img)
                prob_gang_t = self.discriminator(decoded_svhn_img)

                gang_t_loss = F.nll_loss(prob_gang_t, labels_3)

                gang_loss = gang_s_loss + gang_t_loss

                # const_loss
                encoded_mnist_img = self.encoder(mnist_img)

                encoded_mnist_img = self.encoder(mnist_img)
                decoded_mnist_img = self.decoder(encoded_mnist_img)
                encoded_mnist_img_const = self.encoder(decoded_mnist_img)

                const_loss = F.mse_loss(
                    encoded_mnist_img, encoded_mnist_img_const)

                # tid_loss
                encoded_svhn_img = self.encoder(svhn_img)
                decoded_svhn_img = self.decoder(encoded_svhn_img)

                tid_loss = F.mse_loss(svhn_img, decoded_svhn_img)

                # tv_loss
                encoded_svhn_img = self.encoder(svhn_img)
                decoded_svhn_img = self.decoder(encoded_svhn_img)
                shift_0_img = decoded_svhn_img[1:, :]
                shift_0_img_vector = decoded_svhn_img[-1, :]
                be_tensor_0 = shift_0_img_vector.unsqueeze(0)
                generated_img_0 = torch.cat((shift_0_img, be_tensor_0), dim=0)
                i_gap = F.mse_loss(decoded_svhn_img, generated_img_0)
                shift_1_img = decoded_svhn_img[:, 1:]
                shift_1_img_vector = decoded_svhn_img[:, -1]
                be_tensor_1 = shift_1_img_vector.unsqueeze(1)
                generated_img_1 = torch.cat((shift_1_img, be_tensor_1), dim=1)
                j_gap = F.mse_loss(decoded_svhn_img, generated_img_1)
                tv_loss = torch.pow(
                    (i_gap + j_gap),
                    torch.tensor(
                        [0.5]).to(
                        self.device))

                # generator_loss
                a = 100
                b = 1
                c = 0.05

                generator_loss = gang_loss + a * const_loss + b * \
                    tid_loss + c * tv_loss
                generator_loss.backward()
                self.encoder_optim.step()
                self.decoder_optim.step()
            print("epoch: {} L(D(x)): {} L(G(x)): {}".format(
                e, discriminator_loss, generator_loss))
            torchvision.utils.save_image(
                decoded_svhn_img, "results/{}_result.jpg".format(e))
