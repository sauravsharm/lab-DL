import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

import wandb

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

transform_original = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transform_augmented = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

full_dataset_original = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform_original
)

full_dataset_augmented = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform_augmented
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_val_transform
)

train_size = int(0.8 * len(full_dataset_original))   # 40000
val_size = len(full_dataset_original) - train_size   # 10000

train_dataset_original, val_dataset_original = random_split(
    full_dataset_original,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_dataset_augmented, _ = random_split(
    full_dataset_augmented,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

batch_size = 128

train_loader_original = DataLoader(train_dataset_original, batch_size=batch_size, shuffle=True)
train_loader_augmented = DataLoader(train_dataset_augmented, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset_original, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

def imshow(img):
    img = img.permute(1, 2, 0).numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis("off")

images, labels = next(iter(train_loader_original))

plt.figure(figsize=(10, 5))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    imshow(images[i])
    plt.title(classes[labels[i]])
plt.tight_layout()
plt.show()

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)                 # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)                # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)           # [B, num_patches, embed_dim]
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=128,
        depth=6,
        num_heads=4,
        mlp_dim=256,
        dropout=0.1
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x)

        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits

def get_resnet18(num_classes=10):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_cross_entropy_loss():
    return nn.CrossEntropyLoss()

def get_label_smoothing_loss(smoothing=0.1):
    return nn.CrossEntropyLoss(label_smoothing=smoothing)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
    
def get_loss_function(loss_name="cross_entropy"):
    if loss_name == "cross_entropy":
        return get_cross_entropy_loss()
    elif loss_name == "label_smoothing":
        return get_label_smoothing_loss()
    elif loss_name == "focal":
        return FocalLoss()
    else:
        raise ValueError("Invalid loss function name")
    
def get_optimizer(model, optimizer_name="adam", lr=1e-3):
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer name")
    
def calculate_accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10):
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_acc += calculate_accuracy(outputs, labels) * images.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_acc += calculate_accuracy(outputs, labels) * images.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    return history

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            test_acc += calculate_accuracy(outputs, labels) * images.size(0)

    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)

    return test_loss, test_acc

def plot_history(history, title="Training History"):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title(title + " - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.title(title + " - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

vit_model = VisionTransformer(
    img_size=32,
    patch_size=4,
    embed_dim=128,
    depth=6,
    num_heads=4,
    mlp_dim=256,
    num_classes=10
)

criterion = get_loss_function("cross_entropy")
optimizer = get_optimizer(vit_model, "adam", lr=1e-3)

vit_history = train_model(
    vit_model,
    train_loader_original,
    val_loader,
    optimizer,
    criterion,
    epochs=10
)

plot_history(vit_history, title="ViT")

resnet_model = get_resnet18(num_classes=10)

criterion = get_loss_function("cross_entropy")
optimizer = get_optimizer(resnet_model, "adam", lr=1e-3)

resnet_history = train_model(
    resnet_model,
    train_loader_original,
    val_loader,
    optimizer,
    criterion,
    epochs=10
)

plot_history(resnet_history, title="ResNet-18")

vit_test_loss, vit_test_acc = evaluate_model(vit_model, test_loader, criterion)
resnet_test_loss, resnet_test_acc = evaluate_model(resnet_model, test_loader, criterion)

print("ViT Test Loss:", vit_test_loss)
print("ViT Test Accuracy:", vit_test_acc)

print("ResNet-18 Test Loss:", resnet_test_loss)
print("ResNet-18 Test Accuracy:", resnet_test_acc)

model_types = ["vit", "resnet18"]
dataset_types = ["original", "augmented"]
loss_functions = ["cross_entropy", "label_smoothing", "focal"]
optimizers_list = ["sgd", "rmsprop", "adam"]

results = []

for model_type in model_types:
    for dataset_type in dataset_types:
        for loss_name in loss_functions:
            for opt_name in optimizers_list:
                print(f"\nRunning: {model_type}, {dataset_type}, {loss_name}, {opt_name}")

                if model_type == "vit":
                    model = VisionTransformer(
                        img_size=32,
                        patch_size=4,
                        embed_dim=128,
                        depth=6,
                        num_heads=4,
                        mlp_dim=256,
                        num_classes=10
                    )
                else:
                    model = get_resnet18(num_classes=10)

                train_loader = train_loader_original if dataset_type == "original" else train_loader_augmented
                criterion = get_loss_function(loss_name)
                optimizer = get_optimizer(model, opt_name, lr=1e-3)

                start_time = time.time()

                history = train_model(
                    model,
                    train_loader,
                    val_loader,
                    optimizer,
                    criterion,
                    epochs=5
                )

                training_time = time.time() - start_time
                test_loss, test_acc = evaluate_model(model, test_loader, criterion)

                results.append({
                    "model": model_type,
                    "dataset": dataset_type,
                    "loss_function": loss_name,
                    "optimizer": opt_name,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "training_time_sec": training_time
                })

import pandas as pd

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("vit_resnet_results.csv", index=False)

torch.save(vit_model.state_dict(), "vit_model.pth")
torch.save(resnet_model.state_dict(), "resnet18_model.pth")

from huggingface_hub import create_repo, upload_file

create_repo("dl-lab10-vit-resnet", private=False, exist_ok=True)

upload_file(
    path_or_fileobj="vit_model.pth",
    path_in_repo="vit_model.pth",
    repo_id="sosososorav/dl-lab10-vit-resnet"
)

upload_file(
    path_or_fileobj="resnet18_model.pth",
    path_in_repo="resnet18_model.pth",
    repo_id="sosososorav/dl-lab10-vit-resnet"
)

def train_classifier_wandb(model, train_loader, val_loader, optimizer, criterion, config):
    wandb.init(entity="sharmasaurav1319_dtu", project="cifar10-vit-resnet", config=config)

    epochs = config["epochs"]

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_acc += calculate_accuracy(outputs, labels) * images.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_acc += calculate_accuracy(outputs, labels) * images.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    wandb.finish()

model_types = ["vit", "resnet18"]
dataset_types = ["original", "augmented"]
loss_functions = ["cross_entropy", "label_smoothing", "focal"]
optimizers_list = ["sgd", "rmsprop", "adam"]

for model_type in model_types:
    for dataset_type in dataset_types:
        for loss_name in loss_functions:
            for opt_name in optimizers_list:
                if model_type == "vit":
                    model = VisionTransformer(
                        img_size=32,
                        patch_size=4,
                        embed_dim=128,
                        depth=6,
                        num_heads=4,
                        mlp_dim=256,
                        num_classes=10
                    ).to(device)
                else:
                    model = get_resnet18(num_classes=10).to(device)

                train_loader = train_loader_original if dataset_type == "original" else train_loader_augmented
                criterion = get_loss_function(loss_name)
                optimizer = get_optimizer(model, opt_name, lr=1e-3)

                config = {
                    "model": model_type,
                    "dataset": dataset_type,
                    "loss_function": loss_name,
                    "optimizer": opt_name,
                    "epochs": 5,
                    "lr": 1e-3
                }

                train_classifier_wandb(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    config=config
                )