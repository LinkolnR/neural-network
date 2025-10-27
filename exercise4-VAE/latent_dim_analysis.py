import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, latent_dim=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h = self.relu(self.fc1(x.view(-1, 784)))
        h = self.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=20):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        x_recon = self.sigmoid(self.fc3(h))
        return x_recon

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

def vae_loss(x_recon, x, mu, logvar):
    bce = nn.BCELoss(reduction='sum')
    reconstruction_loss = bce(x_recon, x.view(-1, 784))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss, kl_loss

def load_data(batch_size=64):
    train_data = datasets.MNIST(root='./data', train=True, download=True,
                                transform=transforms.ToTensor())
    val_data = datasets.MNIST(root='./data', train=False, download=True,
                              transform=transforms.ToTensor())
    
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_set, _ = random_split(train_data, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_vae(model, train_loader, val_loader, epochs=10, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_recon_losses = []
    train_kl_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_recon_loss = 0
        train_kl_loss = 0
        
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar, _ = model(x)
            recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)
            total_loss = recon_loss + kl_loss
            total_loss.backward()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            optimizer.step()
        
        avg_recon = train_recon_loss / len(train_loader.dataset)
        avg_kl = train_kl_loss / len(train_loader.dataset)
        train_recon_losses.append(avg_recon)
        train_kl_losses.append(avg_kl)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_recon, mu, logvar, _ = model(x)
                recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)
                val_loss += (recon_loss + kl_loss).item()
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
    
    return train_recon_losses, train_kl_losses, val_losses

def visualize_latent_dim_effect(latent_dims, results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    final_recon = [r[0][-1] for r in results]
    final_kl = [r[1][-1] for r in results]
    final_val = [r[2][-1] for r in results]
    
    axes[0].plot(latent_dims, final_recon, marker='o', linewidth=2)
    axes[0].set_xlabel('Latent Dimensions')
    axes[0].set_ylabel('Final Reconstruction Loss')
    axes[0].set_title('Reconstruction Loss vs Latent Dim')
    axes[0].grid(True)
    
    axes[1].plot(latent_dims, final_kl, marker='s', linewidth=2, color='orange')
    axes[1].set_xlabel('Latent Dimensions')
    axes[1].set_ylabel('Final KL Loss')
    axes[1].set_title('KL Divergence vs Latent Dim')
    axes[1].grid(True)
    
    axes[2].plot(latent_dims, final_val, marker='^', linewidth=2, color='green')
    axes[2].set_xlabel('Latent Dimensions')
    axes[2].set_ylabel('Final Validation Loss')
    axes[2].set_title('Total Loss vs Latent Dim')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('./results/latent_dim_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_samples_by_latent_dim(model, latent_dim, num_samples=9):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decoder(z).cpu().numpy()
        
        grid_size = 3
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))
        for idx, ax in enumerate(axes.flat):
            if idx < num_samples:
                ax.imshow(samples[idx].reshape(28, 28), cmap='gray')
            ax.axis('off')
        
        return fig

if __name__ == '__main__':
    os.makedirs('./results', exist_ok=True)
    
    print('Loading data...')
    train_loader, val_loader = load_data(batch_size=64)
    
    latent_dims = [2, 5, 10, 20, 50]
    results = []
    
    print('Training VAEs with different latent dimensions...')
    for latent_dim in latent_dims:
        print(f'\nTraining VAE with latent_dim={latent_dim}')
        model = VAE(latent_dim=latent_dim).to(device)
        recon, kl, val = train_vae(model, train_loader, val_loader, epochs=10)
        results.append((recon, kl, val))
        
        fig = generate_samples_by_latent_dim(model, latent_dim)
        plt.savefig(f'./results/samples_latent_dim_{latent_dim}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print('\nGenerating analysis plots...')
    visualize_latent_dim_effect(latent_dims, results)
    
    print('Done! Analysis saved in ./results/')
