import os
import torch
import random
import torchvision.transforms
from torch.utils.data import sampler
from torch.utils.data.dataloader import DataLoader, RandomSampler
from torch.nn.modules.loss import BCELoss
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.utils import make_grid, save_image
import logging
from torch import nn
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim
import numpy as np

logger = logging.getLogger()
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = BCELoss(reduction="sum")
        self.kl = lambda mu, logvar : torch.sum(-logvar + torch.exp(logvar) + mu ** 2 - 1) / 2
    
    def forward(self, recons, x, mu, logvar):
        L_recons = self.bce(recons, x)
        L_kl = self.kl(mu, logvar)
        return L_recons + L_kl

class MNISTCVAE(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim_1=512, hidden_dim_2=256, latent_dim=64, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc_input_hidden = nn.Linear(self.input_dim + self.num_classes, self.hidden_dim_1)
        self.fc_hidden_hidden_1 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.fc_mu = nn.Linear(self.hidden_dim_2, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim_2, self.latent_dim)

        self.fc_z_hidden = nn.Linear(self.latent_dim + self.num_classes, self.hidden_dim_1)
        self.fc_hidden_hidden_2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.fc_output = nn.Linear(self.hidden_dim_2, self.input_dim)

    def forward(self, x, c):
        mu, logvar = self.encode(x.reshape(-1, 28 * 28), c)
        z = self.reparam(mu, logvar)
        return self.decode(z, c).reshape(x.size()), mu, logvar

    def encode(self, x, c):
        xc = torch.cat([x, c], dim=1) # B x (D + C)
        hidden = self.relu(self.fc_input_hidden(xc))
        hidden = self.relu(self.fc_hidden_hidden_1(hidden))
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def decode(self, z, c):
        zc = torch.cat([z, c], dim=1)
        hidden = self.relu(self.fc_z_hidden(zc))
        hidden = self.relu(self.fc_hidden_hidden_2(hidden))
        return self.sigmoid(self.fc_output(hidden))

    def reparam(self, mu, logvar):
        std = torch.exp(logvar / 2)
        z = torch.randn_like(logvar)
        return mu + z * std

class MNISTConvCVAE(nn.Module):
    def __init__(self, latent_dim=64, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.conv_1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=7), nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(16), nn.ReLU()) # B x 16 x 11 x 11
        self.conv_2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5), nn.BatchNorm2d(32), nn.ReLU()) # B x 32 x 7 x 7

        self.fc_1 = nn.Sequential(nn.Linear(32 * 7 * 7, 1024), nn.ReLU())
        self.fc_2 = nn.Sequential(nn.Linear(1024 + self.num_classes, 512), nn.ReLU())
        self.fc_3 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.fc_mu = nn.Linear(512, self.latent_dim)
        self.fc_logvar = nn.Linear(512, self.latent_dim)

        self.fc_4 = nn.Sequential(nn.Linear(self.latent_dim + self.num_classes, 1024), nn.ReLU())
        self.fc_5 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.fc_6 = nn.Sequential(nn.Linear(512, 32 * 14 * 14), nn.ReLU())
        self.up_conv_1 = nn.Sequential(nn.ConvTranspose2d(32, 32, 3), nn.BatchNorm2d(32), nn.ReLU())
        self.up_conv_2 = nn.Sequential(nn.ConvTranspose2d(32, 16, 7), nn.BatchNorm2d(16), nn.ReLU())
        self.up_conv_3 = nn.ConvTranspose2d(16, 1, 7)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparam(mu, logvar)
        return self.decode(z, c), mu, logvar

    def encode(self, x, c):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.fc_1(x.reshape(-1, 32 * 7 * 7))
        xc = torch.cat([x, c], dim=1) # B x (D + C)
        hidden = self.fc_2(xc)
        hidden = self.fc_3(hidden)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def decode(self, z, c):
        zc = torch.cat([z, c], dim=1)
        hidden = self.fc_6(self.fc_5(self.fc_4(zc)))
        return self.sigmoid(self.up_conv_3(self.up_conv_2(self.up_conv_1(hidden.reshape(z.size(0), 32, 14, 14)))))

    def reparam(self, mu, logvar):
        std = torch.exp(logvar / 2)
        z = torch.randn_like(logvar)
        return mu + z * std

DATASETS = {
    "mnist": MNIST,
    "fashionmnist": FashionMNIST,
}

MODELS = {
    "linear": MNISTCVAE,
    "conv": MNISTConvCVAE,
}

if __name__ == "__main__":    
    parser = ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--model", type=str, default="linear")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--image_dir", type=str, default="images")
    parser.add_argument("--seed", type=int, default=315)

    args = parser.parse_args()

    logger.info(f"Training args: {args}")

    train_dataset = FashionMNIST("./", download=True, transform=torchvision.transforms.ToTensor())
    logger.info(f"Loaded {args.dataset} train dataset")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed)

    sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    model = MNISTConvCVAE(latent_dim=args.latent_dim).to(device)
    loss_fn = VAELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.zero_grad()
    eval_z = torch.randn((100, args.latent_dim)).to(device)
    eval_labels = torch.zeros((100, ), dtype=torch.long).to(device)
    for i in range(100):
        eval_labels[i] = i // 10
    eval_labels = F.one_hot(eval_labels, num_classes=10)
    def evaluate(show=False, epoch=0):
        model.eval()
        
        with torch.no_grad():
            reconstruction = model.decode(eval_z, eval_labels).reshape(100, 1, 28, 28)
            if show:
                grid = make_grid(reconstruction.detach().cpu(), nrow=10).numpy()
                plt.imshow(grid.transpose(1, 2, 0), cmap="gray")
                plt.show()
            else:
                if not os.path.exists(args.image_dir):
                    os.makedirs(args.image_dir)
                save_image(reconstruction, f"{args.image_dir}/eval_{epoch}.png", nrow=10)

    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        tr_loss = 0
        tr_step = 0
        with tqdm(train_loader) as t:
            for batch in t:
                images, labels = (x.to(device) for x in batch)
                labels = F.one_hot(labels, num_classes=10)
                reconstruction, mu, logvar = model(images, labels)
                loss = loss_fn(reconstruction, images, mu, logvar)
                loss.backward()
                optimizer.step()
                model.zero_grad()

                tr_loss += loss.item()
                tr_step += 1

                t.set_postfix({"loss": tr_loss / tr_step})
        evaluate(epoch=epoch)
    
    evaluate(show=True)
    
    
