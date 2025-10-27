import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

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

class StandardAutoencoder(nn.Module):
    def __init__(self, latent_dim=20):
        super(StandardAutoencoder, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim)
        self.fc4 = nn.Linear(latent_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.latent_dim = latent_dim
    
    def forward(self, x):
        h = self.relu(self.fc1(x.view(-1, 784)))
        h = self.relu(self.fc2(h))
        z = self.relu(self.fc3(h))
        h = self.relu(self.fc4(z))
        h = self.relu(self.fc5(h))
        x_recon = self.sigmoid(self.fc6(h))
        return x_recon, z

def load_data(batch_size=64):
    train_data = datasets.MNIST(root='./data', train=True, download=True,
                                transform=transforms.ToTensor())
    test_data = datasets.MNIST(root='./data', train=False, download=True,
                               transform=transforms.ToTensor())
    
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_set, val_set = random_split(train_data, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def vae_loss(x_recon, x, mu, logvar):
    bce = nn.BCELoss(reduction='sum')
    reconstruction_loss = bce(x_recon, x.view(-1, 784))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_loss

def train_vae(model, train_loader, val_loader, epochs=20, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar, _ = model(x)
            loss = vae_loss(x_recon, x, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_recon, mu, logvar, _ = model(x)
                loss = vae_loss(x_recon, x, mu, logvar)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'VAE Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

def train_autoencoder(model, train_loader, val_loader, epochs=20, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss(reduction='mean')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, _ = model(x)
            loss = loss_fn(x_recon, x.view(-1, 784))
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_recon, _ = model(x)
                loss = loss_fn(x_recon, x.view(-1, 784))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'AE Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

def compare_reconstructions(vae_model, ae_model, test_loader, num_images=5):
    vae_model.eval()
    ae_model.eval()
    
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x.to(device)
        
        vae_recon, _, _, _ = vae_model(x)
        ae_recon, _ = ae_model(x)
        
        x = x.cpu().numpy()
        vae_recon = vae_recon.cpu().numpy()
        ae_recon = ae_recon.cpu().numpy()
        
        fig, axes = plt.subplots(3, num_images, figsize=(15, 9))
        for i in range(num_images):
            axes[0, i].imshow(x[i].reshape(28, 28), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(vae_recon[i].reshape(28, 28), cmap='gray')
            axes[1, i].set_title('VAE Recon')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(ae_recon[i].reshape(28, 28), cmap='gray')
            axes[2, i].set_title('AE Recon')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('./results/comparison_reconstructions.png', dpi=150, bbox_inches='tight')
        plt.close()

def compare_latent_spaces(vae_model, ae_model, test_loader):
    vae_model.eval()
    ae_model.eval()
    
    vae_latent = []
    ae_latent = []
    labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            _, _, _, vae_z = vae_model(x)
            _, ae_z = ae_model(x)
            
            vae_latent.append(vae_z.cpu().numpy())
            ae_latent.append(ae_z.cpu().numpy())
            labels.append(y.numpy())
    
    vae_latent = np.concatenate(vae_latent)
    ae_latent = np.concatenate(ae_latent)
    labels = np.concatenate(labels)
    
    tsne = TSNE(n_components=2, random_state=42)
    vae_2d = tsne.fit_transform(vae_latent)
    ae_2d = tsne.fit_transform(ae_latent)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scatter1 = axes[0].scatter(vae_2d[:, 0], vae_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=30)
    axes[0].set_title('VAE Latent Space')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    
    scatter2 = axes[1].scatter(ae_2d[:, 0], ae_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=30)
    axes[1].set_title('Autoencoder Latent Space')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    
    plt.colorbar(scatter2, ax=axes[1], label='Digit')
    plt.tight_layout()
    plt.savefig('./results/comparison_latent_spaces.png', dpi=150, bbox_inches='tight')
    plt.close()

def compare_losses(vae_train, vae_val, ae_train, ae_val):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(vae_train, label='VAE Train', linewidth=2)
    axes[0].plot(vae_val, label='VAE Val', linewidth=2)
    axes[0].set_title('VAE Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(ae_train, label='AE Train', linewidth=2)
    axes[1].plot(ae_val, label='AE Val', linewidth=2)
    axes[1].set_title('Autoencoder Training Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('./results/comparison_losses.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    import os
    os.makedirs('./results', exist_ok=True)
    
    print('Loading data...')
    train_loader, val_loader, test_loader = load_data(batch_size=64)
    
    print('Training VAE...')
    vae = VAE(latent_dim=20).to(device)
    vae_train_losses, vae_val_losses = train_vae(vae, train_loader, val_loader, epochs=20)
    
    print('Training Autoencoder...')
    ae = StandardAutoencoder(latent_dim=20).to(device)
    ae_train_losses, ae_val_losses = train_autoencoder(ae, train_loader, val_loader, epochs=20)
    
    print('Generating comparison visualizations...')
    compare_reconstructions(vae, ae, test_loader, num_images=5)
    compare_latent_spaces(vae, ae, test_loader)
    compare_losses(vae_train_losses, vae_val_losses, ae_train_losses, ae_val_losses)
    
    print('Done! Comparison saved in ./results/')
