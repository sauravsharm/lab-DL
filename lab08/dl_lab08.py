import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from sklearn.manifold import TSNE

import wandb

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.ToTensor()

full_train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset_full = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

full_dataset = torch.utils.data.ConcatDataset([full_train_dataset, test_dataset_full])

total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train: {len(train_dataset)}")
print(f"Val:   {len(val_dataset)}")
print(f"Test:  {len(test_dataset)}")

def show_batch(loader):
    images, labels = next(iter(loader))
    grid = make_grid(images[:16], nrow=4)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.title("Sample Fashion-MNIST Images")
    plt.show()

show_batch(train_loader)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        x_recon = x_recon.view(-1, 1, 28, 28)
        return x_recon

class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x_recon = self.decoder(z)
        return x_recon.view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def ae_loss_function(recon_x, x, loss_type="bce"):
    if loss_type == "bce":
        return F.binary_cross_entropy(recon_x, x, reduction="mean")
    elif loss_type == "mse":
        return F.mse_loss(recon_x, x, reduction="mean")
    else:
        raise ValueError("loss_type must be 'bce' or 'mse'")


def vae_loss_function(recon_x, x, mu, logvar, loss_type="bce", beta=1.0):
    if loss_type == "bce":
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="mean")
    elif loss_type == "mse":
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    else:
        raise ValueError("loss_type must be 'bce' or 'mse'")
    
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss

def get_optimizer(model, optimizer_name="adam", lr=1e-3):
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("optimizer_name must be 'sgd', 'rmsprop', or 'adam'")

def train_autoencoder(model, train_loader, val_loader, optimizer, loss_type="bce", epochs=10):
    history = {
        "train_loss": [],
        "val_loss": []
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, _ in train_loader:
            images = images.to(device)
            
            optimizer.zero_grad()
            recon = model(images)
            loss = ae_loss_function(recon, images, loss_type=loss_type)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                recon = model(images)
                loss = ae_loss_function(recon, images, loss_type=loss_type)
                val_loss += loss.item() * images.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    return history

def train_vae(model, train_loader, val_loader, optimizer, loss_type="bce", epochs=10, beta=1.0):
    history = {
        "train_total_loss": [],
        "train_recon_loss": [],
        "train_kl_loss": [],
        "val_total_loss": [],
        "val_recon_loss": [],
        "val_kl_loss": []
    }
    
    for epoch in range(epochs):
        model.train()
        train_total = 0.0
        train_recon = 0.0
        train_kl = 0.0
        
        for images, _ in train_loader:
            images = images.to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = vae_loss_function(
                recon, images, mu, logvar, loss_type=loss_type, beta=beta
            )
            loss.backward()
            optimizer.step()
            
            train_total += loss.item() * images.size(0)
            train_recon += recon_loss.item() * images.size(0)
            train_kl += kl_loss.item() * images.size(0)
        
        train_total /= len(train_loader.dataset)
        train_recon /= len(train_loader.dataset)
        train_kl /= len(train_loader.dataset)
        
        model.eval()
        val_total = 0.0
        val_recon = 0.0
        val_kl = 0.0
        
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                recon, mu, logvar = model(images)
                loss, recon_loss, kl_loss = vae_loss_function(
                    recon, images, mu, logvar, loss_type=loss_type, beta=beta
                )
                
                val_total += loss.item() * images.size(0)
                val_recon += recon_loss.item() * images.size(0)
                val_kl += kl_loss.item() * images.size(0)
        
        val_total /= len(val_loader.dataset)
        val_recon /= len(val_loader.dataset)
        val_kl /= len(val_loader.dataset)
        
        history["train_total_loss"].append(train_total)
        history["train_recon_loss"].append(train_recon)
        history["train_kl_loss"].append(train_kl)
        history["val_total_loss"].append(val_total)
        history["val_recon_loss"].append(val_recon)
        history["val_kl_loss"].append(val_kl)
        
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Total: {train_total:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f} | "
            f"Val Total: {val_total:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}"
        )
    
    return history

def plot_history(history, title="Training History"):
    plt.figure(figsize=(8, 5))
    for key, values in history.items():
        plt.plot(values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def show_reconstructions(model, loader, model_type="ae", n=8):
    model.eval()
    images, _ = next(iter(loader))
    images = images[:n].to(device)
    
    with torch.no_grad():
        if model_type == "ae":
            recon = model(images)
        else:
            recon, _, _ = model(images)
    
    images = images.cpu()
    recon = recon.cpu()
    
    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        axes[0, i].imshow(images[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
    
    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Recon")
    plt.tight_layout()
    plt.show()

def generate_from_vae(model, num_samples=16):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.fc_mu.out_features).to(device)
        samples = model.decode(z).cpu()
    
    grid = make_grid(samples, nrow=4)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.title("Generated Samples from VAE")
    plt.show()

def extract_latent_vectors(model, loader, model_type="ae"):
    model.eval()
    latent_vectors = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            
            if model_type == "ae":
                z = model.encoder(images)
            else:
                mu, logvar = model.encode(images)
                z = mu
            
            latent_vectors.append(z.cpu())
            labels_list.append(labels)
    
    latent_vectors = torch.cat(latent_vectors, dim=0).numpy()
    labels_list = torch.cat(labels_list, dim=0).numpy()
    
    return latent_vectors, labels_list

def plot_latent_space(latent_vectors, labels, title="Latent Space"):
    if latent_vectors.shape[1] == 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            latent_vectors[:, 0], latent_vectors[:, 1],
            c=labels, cmap="tab10", s=8
        )
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.grid(True)
        plt.show()
    else:
        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(latent_vectors)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            reduced[:, 0], reduced[:, 1],
            c=labels, cmap="tab10", s=8
        )
        plt.colorbar(scatter)
        plt.title(title + " (t-SNE)")
        plt.grid(True)
        plt.show()

def latent_interpolation(model, img1, img2, model_type="ae", steps=10):
    model.eval()
    
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    
    with torch.no_grad():
        if model_type == "ae":
            z1 = model.encoder(img1)
            z2 = model.encoder(img2)
            decoder = model.decoder
        else:
            mu1, _ = model.encode(img1)
            mu2, _ = model.encode(img2)
            z1, z2 = mu1, mu2
            decoder = model.decode
        
        interpolated_images = []
        for alpha in np.linspace(0, 1, steps):
            z = (1 - alpha) * z1 + alpha * z2
            if model_type == "ae":
                recon = decoder(z).view(-1, 1, 28, 28)
            else:
                recon = decoder(z)
            interpolated_images.append(recon.cpu().squeeze(0))
    
    fig, axes = plt.subplots(1, steps, figsize=(2*steps, 2))
    for i, img in enumerate(interpolated_images):
        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"{i}")
    plt.tight_layout()
    plt.show()

def evaluate_autoencoder(model, loader, loss_type="bce"):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            recon = model(images)
            loss = ae_loss_function(recon, images, loss_type=loss_type)
            total_loss += loss.item() * images.size(0)
    
    return total_loss / len(loader.dataset)


def evaluate_vae(model, loader, loss_type="bce", beta=1.0):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = vae_loss_function(
                recon, images, mu, logvar, loss_type=loss_type, beta=beta
            )
            total_loss += loss.item() * images.size(0)
            total_recon += recon_loss.item() * images.size(0)
            total_kl += kl_loss.item() * images.size(0)
    
    total_loss /= len(loader.dataset)
    total_recon /= len(loader.dataset)
    total_kl /= len(loader.dataset)
    
    return total_loss, total_recon, total_kl

latent_dim = 16
loss_type = "bce"
optimizer_name = "adam"
epochs = 10

ae_model = Autoencoder(latent_dim=latent_dim).to(device)
ae_optimizer = get_optimizer(ae_model, optimizer_name=optimizer_name, lr=1e-3)

ae_history = train_autoencoder(
    ae_model,
    train_loader,
    val_loader,
    ae_optimizer,
    loss_type=loss_type,
    epochs=epochs
)

plot_history(ae_history, title="Autoencoder Training")
show_reconstructions(ae_model, test_loader, model_type="ae")

ae_test_loss = evaluate_autoencoder(ae_model, test_loader, loss_type=loss_type)
print("Autoencoder Test Loss:", ae_test_loss)

latent_dim = 16
loss_type = "bce"
optimizer_name = "adam"
epochs = 10

vae_model = VAE(latent_dim=latent_dim).to(device)
vae_optimizer = get_optimizer(vae_model, optimizer_name=optimizer_name, lr=1e-3)

vae_history = train_vae(
    vae_model,
    train_loader,
    val_loader,
    vae_optimizer,
    loss_type=loss_type,
    epochs=epochs,
    beta=1.0
)

plot_history(vae_history, title="VAE Training")
show_reconstructions(vae_model, test_loader, model_type="vae")
generate_from_vae(vae_model)

vae_test_total, vae_test_recon, vae_test_kl = evaluate_vae(
    vae_model,
    test_loader,
    loss_type=loss_type,
    beta=1.0
)

print("VAE Test Total Loss:", vae_test_total)
print("VAE Test Recon Loss:", vae_test_recon)
print("VAE Test KL Loss:", vae_test_kl)

sample_images, sample_labels = next(iter(test_loader))
img1 = sample_images[0]
img2 = sample_images[1]

latent_interpolation(ae_model, img1, img2, model_type="ae", steps=10)
latent_interpolation(vae_model, img1, img2, model_type="vae", steps=10)

latent_dims = [2, 8, 16, 32]
loss_types = ["mse", "bce"]
optimizers_list = ["sgd", "rmsprop", "adam"]

results = []

for latent_dim in latent_dims:
    for loss_type in loss_types:
        for opt_name in optimizers_list:
            print(f"\n=== AE | latent={latent_dim} | loss={loss_type} | opt={opt_name} ===")
            ae_model = Autoencoder(latent_dim=latent_dim).to(device)
            ae_optimizer = get_optimizer(ae_model, optimizer_name=opt_name, lr=1e-3)
            
            train_autoencoder(
                ae_model,
                train_loader,
                val_loader,
                ae_optimizer,
                loss_type=loss_type,
                epochs=5
            )
            
            ae_test_loss = evaluate_autoencoder(ae_model, test_loader, loss_type=loss_type)
            results.append({
                "model": "AE",
                "latent_dim": latent_dim,
                "loss_type": loss_type,
                "optimizer": opt_name,
                "test_loss": ae_test_loss
            })

            print(f"\n=== VAE | latent={latent_dim} | loss={loss_type} | opt={opt_name} ===")
            vae_model = VAE(latent_dim=latent_dim).to(device)
            vae_optimizer = get_optimizer(vae_model, optimizer_name=opt_name, lr=1e-3)
            
            train_vae(
                vae_model,
                train_loader,
                val_loader,
                vae_optimizer,
                loss_type=loss_type,
                epochs=5,
                beta=1.0
            )
            
            vae_test_total, vae_test_recon, vae_test_kl = evaluate_vae(
                vae_model, test_loader, loss_type=loss_type, beta=1.0
            )
            
            results.append({
                "model": "VAE",
                "latent_dim": latent_dim,
                "loss_type": loss_type,
                "optimizer": opt_name,
                "test_loss": vae_test_total,
                "recon_loss": vae_test_recon,
                "kl_loss": vae_test_kl
            })

import pandas as pd

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("ae_vae_results.csv", index=False)

torch.save(ae_model.state_dict(), "autoencoder_latent16.pth")
torch.save(vae_model.state_dict(), "vae_latent16.pth")

def train_autoencoder_wandb(model, train_loader, val_loader, optimizer, config):
    wandb.init(entity="sharmasaurav1319_dtu", project="fashion-mnist-ae-vae", config=config)
    
    loss_type = config["loss_type"]
    epochs = config["epochs"]
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, _ in train_loader:
            images = images.to(device)
            
            optimizer.zero_grad()
            recon = model(images)
            loss = ae_loss_function(recon, images, loss_type=loss_type)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                recon = model(images)
                loss = ae_loss_function(recon, images, loss_type=loss_type)
                val_loss += loss.item() * images.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]['lr']
        })

        if epoch % 2 == 0:
            wandb.log({
                "reconstructions": wandb.Image(recon[0].cpu().numpy())
            })
        
        print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")
    
    wandb.finish()

latent_dims = [2, 8, 16, 32]
loss_types = ["bce", "mse"]
optimizers_list = ["sgd", "rmsprop", "adam"]

for latent_dim in latent_dims:
    for loss_type in loss_types:
        for opt_name in optimizers_list:
            model = Autoencoder(latent_dim=latent_dim).to(device)
            optimizer = get_optimizer(model, optimizer_name=opt_name, lr=1e-3)

            config = {
                "model": "Autoencoder",
                "latent_dim": latent_dim,
                "loss_type": loss_type,
                "optimizer": opt_name,
                "epochs": 5,
                "lr": 1e-3
            }

            train_autoencoder_wandb(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                config=config
            )

from huggingface_hub import create_repo, upload_file

create_repo("dl-lab8-autoencoder", private=False, exist_ok=True)

upload_file(
    path_or_fileobj="autoencoder_latent16.pth",
    path_in_repo="autoencoder_latent16.pth",
    repo_id="sosososorav/dl-lab8-autoencoder"
)

upload_file(
    path_or_fileobj="vae_latent16.pth",
    path_in_repo="vae_latent16.pth",
    repo_id="sosososorav/dl-lab8-autoencoder"
)