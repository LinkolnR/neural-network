import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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
    return reconstruction_loss + kl_loss

def load_data(batch_size=64, dataset='MNIST'):
    if dataset == 'MNIST':
        train_data = datasets.MNIST(root='./data', train=True, download=True,
                                    transform=transforms.ToTensor())
        test_data = datasets.MNIST(root='./data', train=False, download=True,
                                   transform=transforms.ToTensor())
    else:
        train_data = datasets.FashionMNIST(root='./data', train=True, download=True,
                                           transform=transforms.ToTensor())
        test_data = datasets.FashionMNIST(root='./data', train=False, download=True,
                                          transform=transforms.ToTensor())
    
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_set, val_set = random_split(train_data, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_vae(model, train_loader, val_loader, epochs=20, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
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
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

def visualize_reconstructions(model, test_loader, num_images=10, save_path='reconstructions.png'):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x.to(device)
        x_recon, _, _, _ = model(x)
        
        x = x.cpu().numpy()
        x_recon = x_recon.cpu().numpy()
        
        fig, axes = plt.subplots(2, num_images, figsize=(15, 3))
        for i in range(num_images):
            axes[0, i].imshow(x[i].reshape(28, 28), cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(x_recon[i].reshape(28, 28), cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def generate_samples(model, num_samples=16, save_path='generated_samples.png'):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decoder(z).cpu().numpy()
        
        grid_size = int(np.sqrt(num_samples))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
        for idx, ax in enumerate(axes.flat):
            if idx < num_samples:
                ax.imshow(samples[idx].reshape(28, 28), cmap='gray')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def visualize_latent_space(model, test_loader, save_path='latent_space.png'):
    model.eval()
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            _, _, _, z = model(x)
            latent_vectors.append(z.cpu().numpy())
            labels.append(y.numpy())
    
    latent_vectors = np.concatenate(latent_vectors)
    labels = np.concatenate(labels)
    
    if latent_vectors.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_vectors)
    else:
        latent_2d = latent_vectors
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=30)
    plt.colorbar(scatter, label='Digit')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('VAE Latent Space Visualization')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_training_loss(train_losses, val_losses, save_path='training_loss.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    os.makedirs('./results', exist_ok=True)
    
    print('Loading data...')
    train_loader, val_loader, test_loader = load_data(batch_size=64, dataset='MNIST')
    
    print('Initializing VAE...')
    model = VAE(latent_dim=20).to(device)
    
    print('Training VAE...')
    train_losses, val_losses = train_vae(model, train_loader, val_loader, epochs=20, lr=1e-3)
    
    print('Generating visualizations...')
    plot_training_loss(train_losses, val_losses, './results/training_loss.png')
    visualize_reconstructions(model, test_loader, num_images=10, save_path='./results/reconstructions.png')
    generate_samples(model, num_samples=16, save_path='./results/generated_samples.png')
    visualize_latent_space(model, test_loader, save_path='./results/latent_space.png')
    
    torch.save(model.state_dict(), './results/vae_model.pth')
    print('Done! Results saved in ./results/')

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
