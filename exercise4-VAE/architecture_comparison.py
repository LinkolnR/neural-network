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
    def __init__(self, hidden1, hidden2, latent_dim=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_mu = nn.Linear(hidden2, latent_dim)
        self.fc_logvar = nn.Linear(hidden2, latent_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h = self.relu(self.fc1(x.view(-1, 784)))
        h = self.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, hidden1, hidden2, latent_dim=20):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        x_recon = self.sigmoid(self.fc3(h))
        return x_recon

class VAE(nn.Module):
    def __init__(self, hidden1, hidden2, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = Encoder(hidden1, hidden2, latent_dim)
        self.decoder = Decoder(hidden1, hidden2, latent_dim)
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

def train_vae(model, train_loader, val_loader, epochs=20, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    
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
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}')
    
    return train_losses

def get_reconstructions(model, test_loader, num_images=4):
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x.to(device)[:num_images]
        x_recon, _, _, _ = model(x)
        return x.cpu().numpy(), x_recon.cpu().numpy()

def get_generated_samples(model, num_samples=4):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        samples = model.decoder(z).cpu().numpy()
        return samples

# Configurações de arquitetura
configs = {
    'Pequeno (20-40)': {'h1': 20, 'h2': 40},
    'Médio (100-200)': {'h1': 100, 'h2': 200},
    'Grande (200-400)': {'h1': 200, 'h2': 400},
    'Muito Grande (400-800)': {'h1': 400, 'h2': 800}
}

def main():
    os.makedirs('./results', exist_ok=True)
    
    print('Carregando dados...')
    train_loader, val_loader, test_loader = load_data(batch_size=64)
    
    results = {}
    
    for config_name, config in configs.items():
        print(f'\nTreinando {config_name}...')
        model = VAE(hidden1=config['h1'], hidden2=config['h2'], latent_dim=20).to(device)
        
        train_losses = train_vae(model, train_loader, val_loader, epochs=15, lr=1e-3)
        
        x_orig, x_recon = get_reconstructions(model, test_loader, num_images=4)
        x_gen = get_generated_samples(model, num_samples=4)
        
        results[config_name] = {
            'original': x_orig,
            'reconstructed': x_recon,
            'generated': x_gen,
            'losses': train_losses
        }
        
        print(f'{config_name} finalizado!')
    
    # Criar comparação visual em grid 2x2
    fig = plt.figure(figsize=(16, 16))
    config_names = list(results.keys())
    
    for idx, config_name in enumerate(config_names):
        row = idx // 2
        col = idx % 2
        
        # Subplot para reconstruções
        ax1 = plt.subplot(4, 4, idx * 2 + 1)
        config_results = results[config_name]
        
        # Mostrar imagem original e reconstruída lado a lado
        combined = np.zeros((28, 56))
        combined[:, :28] = config_results['original'][0].reshape(28, 28)
        combined[:, 28:] = config_results['reconstructed'][0].reshape(28, 28)
        
        ax1.imshow(combined, cmap='gray')
        ax1.set_title(f'{config_name}\nOriginal | Reconstruído', fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        # Subplot para amostras geradas
        ax2 = plt.subplot(4, 4, idx * 2 + 2)
        ax2.imshow(config_results['generated'][0].reshape(28, 28), cmap='gray')
        ax2.set_title(f'{config_name}\nAmostra Gerada', fontsize=11, fontweight='bold')
        ax2.axis('off')
    
    plt.suptitle('Comparação de Arquitetura VAE: Neurônios e Qualidade', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('./results/architecture_comparison.png', dpi=150, bbox_inches='tight')
    print('\nGráfico de comparação salvo em: ./results/architecture_comparison.png')
    plt.close()
    
    # Criar gráfico de perda para cada configuração
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for config_name, config_results in results.items():
        ax.plot(config_results['losses'], marker='o', label=config_name, linewidth=2)
    
    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel('Perda (Normalizada)', fontsize=12)
    ax.set_title('Convergência por Tamanho de Arquitetura VAE', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./results/architecture_loss_comparison.png', dpi=150, bbox_inches='tight')
    print('Gráfico de perda salvo em: ./results/architecture_loss_comparison.png')
    plt.close()
    
    # Criar análise mais detalhada com 4x4 grid
    fig = plt.figure(figsize=(20, 16))
    
    for idx, config_name in enumerate(config_names):
        config_results = results[config_name]
        
        # Reconstruções
        for i in range(4):
            ax = plt.subplot(4, 8, idx * 8 + i + 1)
            if i == 0:
                ax.imshow(config_results['original'][i].reshape(28, 28), cmap='gray')
            else:
                ax.imshow(config_results['reconstructed'][i].reshape(28, 28), cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_ylabel(config_name, fontsize=12, fontweight='bold')
        
        # Amostras geradas
        for i in range(4):
            ax = plt.subplot(4, 8, idx * 8 + 5 + i)
            ax.imshow(config_results['generated'][i].reshape(28, 28), cmap='gray')
            ax.axis('off')
    
    # Adicionar títulos
    fig.text(0.25, 0.97, 'Reconstruções (Original → Reconstruído)', 
             ha='center', fontsize=14, fontweight='bold')
    fig.text(0.75, 0.97, 'Amostras Geradas', 
             ha='center', fontsize=14, fontweight='bold')
    
    plt.suptitle('Análise Detalhada de Arquitetura VAE', 
                 fontsize=16, fontweight='bold', y=0.99)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('./results/architecture_detailed_comparison.png', dpi=150, bbox_inches='tight')
    print('Gráfico detalhado salvo em: ./results/architecture_detailed_comparison.png')
    plt.close()
    
    print('\nComparação de arquitetura concluída!')
    print('Arquivos gerados:')
    print('  - architecture_comparison.png')
    print('  - architecture_loss_comparison.png')
    print('  - architecture_detailed_comparison.png')

if __name__ == '__main__':
    main()
