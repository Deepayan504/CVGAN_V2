import torch
import torch.nn as nn

# ============================================================================
# Discriminator Architecture (PatchGAN)
# ============================================================================

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=6, features=[64, 128, 256, 512]):
        """
        PatchGAN discriminator.
        
        Args:
            in_channels: input channels (num_input_months + 2 seasonal + 1 target/generated)
            features: list of feature sizes for each layer
        """
        super().__init__()
        
        layers = []
        
        # Initial layer (no batch norm)
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
                nn.LeakyReLU(0.2)
            )
        )
        
        # Intermediate layers
        in_ch = features[0]
        for feature in features[1:]:
            layers.append(CNNBlock(in_ch, feature, stride=1 if feature == features[-1] else 2))
            in_ch = feature
        
        # Final layer
        layers.append(
            nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


# ============================================================================
# Complete Training Script
# ============================================================================

import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from sst_pix2pix_annual_cycle import (
    Generator, SSTDataset, CombinedLoss, 
    compute_climatology, rollout_prediction, get_seasonal_encoding
)

def train_model(sst_data, months, num_epochs=100, num_input_months=3, 
                batch_size=16, lr=0.0002, lambda_clim=0.1):
    """
    Complete training pipeline.
    
    Args:
        sst_data: numpy array of shape (num_samples, 48, 144)
        months: numpy array of shape (num_samples,) with month indices 0-11
        num_epochs: number of training epochs
        num_input_months: number of previous months to use as input
        batch_size: batch size for training
        lr: learning rate
        lambda_clim: weight for climatology loss
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Compute climatology
    print("Computing climatology...")
    climatology = compute_climatology(sst_data, months)
    
    # Create dataset and dataloader
    print("Creating dataset...")
    train_dataset = SSTDataset(
        sst_data=sst_data,
        months=months,
        num_input_months=num_input_months,
        climatology=climatology
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Initialize models
    in_channels = num_input_months + 2  # SST channels + seasonal encoding
    generator = Generator(in_channels=in_channels, features=128, out_channels=1).to(device)
    discriminator = Discriminator(in_channels=in_channels + 1).to(device)  # +1 for target/generated
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
    
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    
    # Loss functions
    criterion = CombinedLoss(climatology_data=climatology, lambda_clim=lambda_clim, lambda_l1=100.0)
    bce_loss = nn.BCEWithLogitsLoss()
    
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=30, gamma=0.5)
    d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=30, gamma=0.5)
    
    # Training history
    history = {
        'g_loss': [],
        'd_loss': [],
        'l1_loss': [],
        'clim_loss': []
    }
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Input channels: {in_channels}\n")
    
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_l1_loss = 0
        epoch_clim_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            target_months = batch['target_month'].to(device)
            
            current_batch_size = inputs.shape[0]
            
            # ===========================
            # Train Discriminator
            # ===========================
            d_optimizer.zero_grad()
            
            # Real pairs
            real_pair = torch.cat([inputs, targets], dim=1)
            d_real = discriminator(real_pair)
            real_labels = torch.ones_like(d_real) * 0.9  # Label smoothing
            d_real_loss = bce_loss(d_real, real_labels)
            
            # Fake pairs
            fake_outputs = generator(inputs)
            fake_pair = torch.cat([inputs, fake_outputs.detach()], dim=1)
            d_fake = discriminator(fake_pair)
            fake_labels = torch.zeros_like(d_fake) + 0.1  # Label smoothing
            d_fake_loss = bce_loss(d_fake, fake_labels)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()
            
            # ===========================
            # Train Generator
            # ===========================
            g_optimizer.zero_grad()
            
            # Generate fake outputs
            fake_outputs = generator(inputs)
            fake_pair = torch.cat([inputs, fake_outputs], dim=1)
            d_fake = discriminator(fake_pair)
            
            # Adversarial loss (want discriminator to think it's real)
            real_labels = torch.ones_like(d_fake)
            g_adv_loss = bce_loss(d_fake, real_labels)
            
            # Reconstruction + Climatology loss
            g_recon_loss, loss_dict = criterion(fake_outputs, targets, target_months)
            
            # Total generator loss
            g_loss = g_adv_loss + g_recon_loss
            g_loss.backward()
            g_optimizer.step()
            
            # Accumulate losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_l1_loss += loss_dict['l1']
            epoch_clim_loss += loss_dict['climatology']
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs} [{batch_idx}/{len(train_loader)}] '
                      f'G: {g_loss.item():.4f} (L1: {loss_dict["l1"]:.4f}, '
                      f'Clim: {loss_dict["climatology"]:.4f}) D: {d_loss.item():.4f}')
        
        # Update learning rates
        g_scheduler.step()
        d_scheduler.step()
        
        # Average losses for epoch
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_l1_loss = epoch_l1_loss / len(train_loader)
        avg_clim_loss = epoch_clim_loss / len(train_loader)
        
        history['g_loss'].append(avg_g_loss)
        history['d_loss'].append(avg_d_loss)
        history['l1_loss'].append(avg_l1_loss)
        history['clim_loss'].append(avg_clim_loss)
        
        print(f'\n=== Epoch {epoch+1}/{num_epochs} Complete ===')
        print(f'Avg G Loss: {avg_g_loss:.4f} | Avg D Loss: {avg_d_loss:.4f}')
        print(f'Avg L1 Loss: {avg_l1_loss:.4f} | Avg Clim Loss: {avg_clim_loss:.4f}\n')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'history': history,
                'climatology': climatology,
                'num_input_months': num_input_months
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')
            print(f'Checkpoint saved: checkpoint_epoch_{epoch+1}.pth')
    
    return generator, discriminator, history, climatology


def evaluate_annual_cycle(generator, test_sst, test_months, num_input_months=3, climatology=None):
    """
    Evaluate the model's ability to capture annual cycle through rollout.
    
    Args:
        generator: trained generator
        test_sst: test SST data (num_samples, 48, 144)
        test_months: month indices for test data
        num_input_months: number of input months
        climatology: monthly climatology (12, 48, 144)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator.eval()
    
    # Select a starting point
    start_idx = num_input_months
    initial_inputs = torch.FloatTensor(test_sst[start_idx-num_input_months:start_idx])
    initial_month = test_months[start_idx - 1]
    
    # Perform 24-month rollout (2 years)
    predictions = rollout_prediction(generator, initial_inputs, initial_month, num_months=24)
    
    # Compute metrics
    predictions_np = np.array(predictions)  # (24, 48, 144)
    
    # Compare with climatology if available
    if climatology is not None:
        # Get climatology for predicted months
        pred_months = [(initial_month + 1 + i) % 12 for i in range(24)]
        clim_values = np.array([climatology[m] for m in pred_months])
        
        # Compute RMSE against climatology
        rmse = np.sqrt(np.mean((predictions_np - clim_values) ** 2))
        print(f"RMSE vs Climatology: {rmse:.4f}")
        
        # Check annual cycle correlation
        pred_mean_cycle = [predictions_np[i::12].mean() for i in range(12)]
        clim_mean_cycle = [climatology[i].mean() for i in range(12)]
        correlation = np.corrcoef(pred_mean_cycle, clim_mean_cycle)[0, 1]
        print(f"Annual Cycle Correlation: {correlation:.4f}")
    
    return predictions


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_training_history(history):
    """Plot training losses over epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(history['g_loss'])
    axes[0, 0].set_title('Generator Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['d_loss'])
    axes[0, 1].set_title('Discriminator Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['l1_loss'])
    axes[1, 0].set_title('L1 Reconstruction Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(history['clim_loss'])
    axes[1, 1].set_title('Climatology Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved: training_history.png")
    plt.close()


def plot_predictions(predictions, climatology, start_month=0):
    """Plot predicted SST fields for 12 months."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    
    for i in range(12):
        row = i // 4
        col = i % 4
        
        month = (start_month + i) % 12
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Plot prediction
        im = axes[row, col].imshow(predictions[i], cmap='RdBu_r', aspect='auto')
        axes[row, col].set_title(f'{month_names[month]} (Predicted)')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('predicted_annual_cycle.png', dpi=300, bbox_inches='tight')
    print("Prediction plot saved: predicted_annual_cycle.png")
    plt.close()


if __name__ == "__main__":
    # Example usage with dummy data
    print("Creating dummy data for demonstration...")
    
    # Create synthetic SST data with annual cycle
    num_years = 50
    num_samples = num_years * 12
    months = np.array([i % 12 for i in range(num_samples)])
    
    # Create data with annual cycle pattern
    sst_data = np.zeros((num_samples, 48, 144))
    for i in range(num_samples):
        month = months[i]
        # Simple annual cycle: sin wave + noise
        base = 15 + 10 * np.sin(2 * np.pi * month / 12)
        sst_data[i] = base + np.random.randn(48, 144) * 2
    
    print(f"Data shape: {sst_data.shape}")
    print(f"Months shape: {months.shape}")
    
    # Train model
    generator, discriminator, history, climatology = train_model(
        sst_data=sst_data,
        months=months,
        num_epochs=5,  # Use more epochs in practice (50-100)
        num_input_months=3,
        batch_size=16,
        lr=0.0002,
        lambda_clim=0.1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate
    print("\nEvaluating annual cycle...")
    predictions = evaluate_annual_cycle(generator, sst_data, months, num_input_months=3, climatology=climatology)
    
    # Plot predictions
    plot_predictions(predictions[:12], climatology, start_month=0)
    
    print("\nTraining complete!")
