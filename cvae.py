import os
import sys
import torch
import random
from torch.nn.modules.activation import Sigmoid
import torchvision.transforms
from torch.utils.data import sampler
from torch.utils.data.dataloader import DataLoader, RandomSampler
from torch.nn.modules.loss import BCELoss, MSELoss
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
import torch.onnx

logger = logging.getLogger()
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

def reparameterize(mu, logvar):
    std = torch.exp(logvar / 2)
    z = torch.randn_like(logvar)
    return mu + z * std

class CVQVAECodebook(nn.Module):
    def __init__(self, num_embeddings=320, latent_dim=3):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(self.num_embeddings, self.latent_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, -1 / self.num_embeddings)
    
    def forward(self, x):
        # x is B x DIM
        # embedding is K x DIM
        distances = (torch.sum(x ** 2, dim=1, keepdim=True) # B x 1
         + torch.sum(self.embedding.weight ** 2, dim=1) # K
         # This will broadcast to B x K
         - 2 * torch.matmul(x, self.embedding.weight.t()) # B x DIM * DIM x K => B x K
        ) # B x K

        codebook_indices = torch.argmin(distances, dim=1).long()
        z_q = self.embedding(codebook_indices)

        return z_q

class MNISTCVQVAE(nn.Module):
    def __init__(self, encoder, decoder, codebook, latent_dim=3, embed_dim=256):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.codebook = codebook

        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        self.fc = nn.Linear(self.embed_dim, self.latent_dim)

    def forward(self, x, c):
        z_e = self.fc(self.encoder(x, c))
        z_q = self.codebook(z_e)
        return self.decoder(z_q, c), z_e, z_q

class VQVAELoss(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.bce = BCELoss(reduction="sum")
        self.beta = beta
        self.mse = MSELoss(reduction="sum")
        
    def forward(self, x, recons, z_e, z_q):
        L_recons = self.bce(recons, x)
        L_embed = self.mse(z_e.detach(), z_q)
        L_encoder = self.mse(z_e, z_q.detach())
        return L_recons + L_embed + L_encoder * self.beta

class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = BCELoss(reduction="sum")
        self.kl = lambda mu, logvar : torch.sum(-logvar + torch.exp(logvar) + mu ** 2 - 1) / 2
    
    def forward(self, x, recons, mu, logvar):
        L_recons = self.bce(recons, x)
        L_kl = self.kl(mu, logvar)
        return L_recons + L_kl

class VAEParamters(nn.Module):
    def __init__(self, embed_dim=256, latent_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.fc_mu = nn.Linear(self.embed_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.embed_dim, self.latent_dim)
    
    def forward(self, x):
        return self.fc_mu(x), self.fc_logvar(x) 

class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim_1=512, hidden_dim_2=256, latent_dim=64, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim + self.num_classes, self.hidden_dim_1),
            nn.ReLU(),

            nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
            nn.ReLU(),
            
            nn.Linear(self.hidden_dim_2, 28 * 28),
            nn.Sigmoid(),
        )
    
    def forward(self, z, c):
        zc = torch.cat([z, c], dim=1)
        return self.fc(zc).reshape(-1, 1, 28, 28)

class MLPEncoder(nn.Module):
    def __init__(self, hidden_dim_1=512, hidden_dim_2=256, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Sequential(
            nn.Linear(28 * 28 + self.num_classes, self.hidden_dim_1),
            nn.ReLU(),

            nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
            nn.ReLU(),
        )

    def forward(self, x, c):
        xc = torch.cat([x.reshape(-1, 28 * 28), c], dim=1)
        hidden = self.fc(xc)
        return hidden

class ConvEncoder(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7), 
            nn.MaxPool2d(kernel_size=2), 
            nn.BatchNorm2d(16), 
            nn.ReLU(), # B x 16 x 11 x 11

            nn.Conv2d(16, 32, kernel_size=5), 
            nn.BatchNorm2d(32),
            nn.ReLU() # B x 32 x 7 x 7
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7 + self.num_classes, 256), 
            nn.ReLU(),

            nn.Linear(256, 512), 
            nn.ReLU(),

            nn.Linear(512, 512), 
            nn.ReLU()
        )
    
    def forward(self, x, c):
        x = self.conv(x)
        x = x.reshape(-1, 32 * 7 * 7)
        xc = torch.cat([x, c], dim=1) # B x (D + C)
        hidden = self.fc(xc)
        return hidden

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=64, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim + self.num_classes, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Linear(512, 32 * 14 * 14), 
            nn.ReLU()
        )

        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 7), 
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 7)
        )

    def forward(self, z, c):
        zc = torch.cat([z, c], dim=1)
        hidden = self.fc(zc)
        return self.sigmoid(self.up_conv(hidden.reshape(-1, 32, 14, 14)))

class MNISTCVAE(nn.Module):
    def __init__(self, encoder, decoder, param):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.param = param

    def forward(self, x, c):
        embed = self.encoder(x, c)
        mu, logvar = self.param(embed)
        z = reparameterize(mu, logvar)
        return self.decoder(z, c), mu, logvar

DATASETS = {
    "mnist": MNIST,
    "fashionmnist": FashionMNIST,
}

ENCODERS = {
    "mlp": MLPEncoder,
    "conv": ConvEncoder,
}

DECODERS = {
    "mlp": MLPDecoder,
    "conv": ConvDecoder,
}

PARAMS = {
    "mlp": {"embed_dim": 256},
    "conv": {"embed_dim": 512},
}

if __name__ == "__main__":    
    parser = ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=3)
    parser.add_argument("--encoder", type=str, choices=ENCODERS.keys(), default="mlp")
    parser.add_argument("--decoder", type=str, choices=DECODERS.keys(), default="mlp")
    parser.add_argument("--dataset", type=str, choices=DATASETS.keys(), default="mnist")
    parser.add_argument("--image_dir", type=str, default="images")
    parser.add_argument("--seed", type=int, default=315)
    parser.add_argument("--export_onnx", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, choices=["cvqvae", "cvae"], default="cvae")

    args = parser.parse_args()
    logger.info(f"Training args: {args}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    encoder = ENCODERS[args.encoder]()
    decoder = DECODERS[args.decoder](latent_dim=args.latent_dim)
    if args.model_type == "cvae":
        param = VAEParamters(**{"latent_dim": args.latent_dim, **PARAMS[args.encoder]})
        model = MNISTCVAE(encoder, decoder, param)
    else:
        codebook = CVQVAECodebook(latent_dim=args.latent_dim)
        model = MNISTCVQVAE(encoder, decoder, codebook, **{"latent_dim": args.latent_dim, **PARAMS[args.encoder]})
    logger.info(f"Model initialized.")

    if args.export_onnx:
        if not args.model_path or not os.path.exists(args.model_path):
            logger.error(f"Cannot find the model at {args.model_path}")
            sys.exit(1)
        model_state_dict = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(model_state_dict)
        logger.info(f"Loaded model {args.model_path}")

        torch.onnx.export(
            model.decoder,
            (torch.rand(10, args.latent_dim), torch.eye(10)),
            "decoder.onnx",
            export_params=True,
            do_constant_folding=True,
            input_names=["z", "c"],
            output_names=["reconstruction"],
            dynamic_axes={
                "z": {0: "batch_size"},
                "c": {0: "batch_size"},
                "reconstruction": {0: "batch_size"},
            }
        )

        logger.info("Exported decoder to decoder.onnx")
        sys.exit(0)

    model = model.to(device)
    train_dataset = DATASETS[args.dataset]("./", download=True, transform=torchvision.transforms.ToTensor())
    logger.info(f"Loaded {args.dataset} train dataset")

    sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)

    loss_fn = VAELoss() if args.model_type == "cvae" else VQVAELoss()
    loss_fn = loss_fn.to(device)
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
            if args.model_type == "cvae":
                reconstruction = model.decoder(eval_z, eval_labels).reshape(100, 1, 28, 28)
            else:
                reconstruction = model.decoder(model.codebook(eval_z), eval_labels).reshape(100, 1, 28, 28)
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
                outputs = model(images, labels)
                loss = loss_fn(images, *outputs)
                loss.backward()
                optimizer.step()
                model.zero_grad()

                tr_loss += loss.item()
                tr_step += 1

                t.set_postfix({"loss": tr_loss / tr_step})
        evaluate(epoch=epoch)
    
    logger.info("Saving model to model.pth")
    torch.save(model.state_dict(), "model.pth")
    torch.save(optimizer.state_dict(), "optimizer.pth")
    logger.info("Saved")
