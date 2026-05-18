import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = 'cuda' #if torch.cuda.is_available() else 'cpu'

# ============================================================================
# STRATEGY 1: Add Seasonal/Temporal Encoding
# ============================================================================

def get_seasonal_encoding(month, shape=(48, 144)):
    """
    Create cyclic encoding for month to preserve annual periodicity.
    
    Args:
        month: int (0-11) representing the month
        shape: tuple (H, W) for the spatial dimensions
    
    Returns:
        torch.Tensor of shape (2, H, W) with sin/cos encoding
    """
    #select a random sample from a normal distribution to add some variability to the encoding
    month += np.random.normal(0, 0.1)  # Add small noise to prevent overfitting to exact month values
    # Cyclic encoding: sin and cos to handle Dec->Jan transition
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    
    # Create spatial maps filled with these values
    sin_channel = torch.full(shape, month_sin, dtype=torch.float32)
    cos_channel = torch.full(shape, month_cos, dtype=torch.float32)
    
    return torch.stack([sin_channel, cos_channel], dim=0)  # (2, 48, 144)


# ============================================================================
# STRATEGY 1: Add Seasonal/Temporal Encoding
# ============================================================================

def get_multiscale_temporal_encoding(sample_idx, total_samples, shape=(48, 144)):
    """
    Create multi-scale temporal encoding to capture variability at different timescales.
    
    This helps the model understand:
    - Position within the year (seasonal cycle)
    - Position within multi-year periods (interannual variability)
    - Long-term trends
    
    Args:
        sample_idx: Current sample index in the dataset
        total_samples: Total number of samples in dataset
        shape: Spatial shape (H, W)
    
    Returns:
        torch.Tensor of shape (6, H, W) with multi-scale encodings
    """
    month = sample_idx % 12
    year_progress = sample_idx / max(total_samples, 1)  # 0 to 1 over entire dataset
    
    # 1. Annual cycle (12-month period)
    annual_sin = np.sin(2 * np.pi * month / 12.0)
    annual_cos = np.cos(2 * np.pi * month / 12.0)
    
    # 2. Interannual cycle (2-7 year periods for ENSO-like variability)
    # ENSO typically has 2-7 year periodicity
    enso_period = 4.0  # years (48 months)
    enso_sin = np.sin(2 * np.pi * sample_idx / (enso_period * 12))
    enso_cos = np.cos(2 * np.pi * sample_idx / (enso_period * 12))
    
    # 3. Decadal variability (10-year periods for PDO-like patterns)
    decadal_period = 10.0  # years (120 months)
    decadal_sin = np.sin(2 * np.pi * sample_idx / (decadal_period * 12))
    
    # 4. Linear trend component (captures long-term warming/cooling)
    trend = year_progress * 2 - 1  # Scale to [-1, 1]
    
    # Create spatial maps filled with these values - ALWAYS 6 channels
    channels = torch.zeros(6, shape[0], shape[1], dtype=torch.float32)
    
    # Fill each channel
    # channels[0, :, :] = annual_sin
    # channels[1, :, :] = annual_cos
    # channels[2, :, :] = enso_sin
    # channels[3, :, :] = enso_cos
    # channels[4, :, :] = decadal_sin
    # channels[5, :, :] = trend
    channels[0, :, :] = enso_sin
    channels[1, :, :] = enso_cos
    channels[2, :, :] = decadal_sin
    
    
    return channels  # Always (3, H, W)



# ============================================================================
# STRATEGY 3: Multi-Month Input Context
# ============================================================================

class SSTDataset(Dataset):
    """
    Dataset for SST with multi-month input and seasonal encoding.
    """
    def __init__(self, sst_data, months, num_input_months=3,climatology=None):
        """
        Args:
            sst_data: numpy array of shape (num_samples, 48, 144) - SST data
            months: numpy array of shape (num_samples,) - month index (0-11) for each sample
            num_input_months: int - number of previous months to use as input
            climatology: numpy array of shape (12, 48, 144) - monthly climatology (optional)
        """
        self.sst_data = sst_data
        self.months = months
        self.num_input_months = num_input_months
        self.climatology = climatology
        
        # Create valid indices (need num_input_months previous samples)
        self.valid_indices = list(range(num_input_months, len(sst_data)))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        
        # Get input: previous num_input_months
        input_sst_list = []
        for i in range(self.num_input_months):
            sst = self.sst_data[actual_idx - self.num_input_months + i]
            input_sst_list.append(torch.FloatTensor(sst).unsqueeze(0))  # (1, 48, 144)
        
        # Stack input months: (num_input_months, 48, 144)
        input_sst = torch.cat(input_sst_list, dim=0)
        
        # Get seasonal encoding for the CURRENT month (the one we're predicting FROM)
        current_month = self.months[actual_idx - 1]
        seasonal_encoding = get_seasonal_encoding(current_month)  # (2, 48, 144)
        
        # Concatenate SST input with seasonal encoding
        # Total input channels: num_input_months + 2
        input_combined = torch.cat([input_sst, seasonal_encoding], dim=0)
        
        # Target: next month's SST
        target_sst = torch.FloatTensor(self.sst_data[actual_idx]).unsqueeze(0)  # (1, 48, 144)
        
        # Get target month for climatology loss
        target_month = self.months[actual_idx]
        
        return {
            'input': input_combined,
            'target': target_sst,
            'current_month': current_month,
            'target_month': target_month
        }




class SSTDataset_new(Dataset):
    """
    Dataset for SST with multi-month input and seasonal encoding.
    """
    def __init__(self, sst_data, months, num_input_months=3,climatology=None):
        """
        Args:
            sst_data: numpy array of shape (num_samples, 48, 144) - SST data
            months: numpy array of shape (num_samples,) - month index (0-11) for each sample
            num_input_months: int - number of previous months to use as input
            climatology: numpy array of shape (12, 48, 144) - monthly climatology (optional)
        """
        self.sst_data = sst_data
        self.months = months
        self.num_input_months = num_input_months
        self.climatology = climatology
        
        # Create valid indices (need num_input_months previous samples)
        self.valid_indices = list(range(num_input_months, len(sst_data)))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        
        # Get input: previous num_input_months
        input_sst_list = []
        for i in range(self.num_input_months):
            sst = self.sst_data[actual_idx - self.num_input_months + i]
            input_sst_list.append(torch.FloatTensor(sst).unsqueeze(0))  # (1, 48, 144)
        
        # Stack input months: (num_input_months, 48, 144)
        input_sst = torch.cat(input_sst_list, dim=0)
        
        # Get seasonal encoding for the CURRENT month (the one we're predicting FROM)
        current_month = self.months[actual_idx - 1]
        seasonal_encoding = get_seasonal_encoding(current_month)  # (2, 48, 144)
        
        # Concatenate SST input with seasonal encoding
        # Total input channels: num_input_months + 2
        #input_combined = torch.cat([input_sst, seasonal_encoding], dim=0)
        
        # Target: next month's SST
        target_sst = torch.FloatTensor(self.sst_data[actual_idx]).unsqueeze(0)  # (1, 48, 144)
        
        # Get target month for climatology loss
        target_month = self.months[actual_idx]
        
        return {
            'input': input_sst,
            'target': target_sst,
            'current_month': current_month,
            'target_month': target_month
        }


# ============================================================================
# STRATEGY 3: Multi-Month Input Context
# ============================================================================

class SSTAnomDataset(Dataset):
    """
    Dataset for SST with multi-month input and seasonal encoding.
    """
    def __init__(self, sst_data, months, num_input_months=3):
        """
        Args:
            sst_data: numpy array of shape (num_samples, 48, 144) - SST data
            months: numpy array of shape (num_samples,) - month index (0-11) for each sample
            num_input_months: int - number of previous months to use as input
            climatology: numpy array of shape (12, 48, 144) - monthly climatology (optional)
        """
        self.sst_data = sst_data
        self.months = months
        self.num_input_months = num_input_months
        #self.climatology = climatology
        
        # Create valid indices (need num_input_months previous samples)
        self.valid_indices = list(range(num_input_months, len(sst_data)))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        
        # Get input: previous num_input_months
        input_sst_list = []
        for i in range(self.num_input_months):
            sst = self.sst_data[actual_idx - self.num_input_months + i]
            input_sst_list.append(torch.FloatTensor(sst).unsqueeze(0))  # (1, 48, 144)
        
        # Stack input months: (num_input_months, 48, 144)
        input_sst = torch.cat(input_sst_list, dim=0)
        
        # Get seasonal encoding for the CURRENT month (the one we're predicting FROM)
        current_month = self.months[actual_idx - 1]
        seasonal_encoding = get_seasonal_encoding(current_month)  # (3, 48, 144)
        
        # Concatenate SST input with seasonal encoding
        # Total input channels: num_input_months + 2
        input_combined = torch.cat([input_sst, seasonal_encoding], dim=0)
        
        # Target: next month's SST
        target_sst = torch.FloatTensor(self.sst_data[actual_idx]).unsqueeze(0)  # (1, 48, 144)
        
        # Get target month for climatology loss
        target_month = self.months[actual_idx]
        
        return {
            'input': input_combined,
            'target': target_sst,
            'current_month': current_month,
            'target_month': target_month
        }
# ============================================================================
# Modified Generator to accept variable input channels
# ============================================================================

class Generator(nn.Module):
    def __init__(self, in_channels=5, features=128, out_channels=1):
        """
        Args:
            in_channels: number of input channels (num_input_months + 2 for seasonal)
            features: base number of features
            out_channels: number of output channels (1 for SST)
        """
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, padding='same'),
            nn.ReLU(0.3),
            nn.Conv2d(features, features, (3,6), (1,3), (1,2), padding_mode="reflect"),
            nn.ReLU(0.3)
        )
        
        # Encoder
        self.down1 = Block(features, features*2, down=True, act="relu", use_dropout=True)
        self.down2 = Block(features*2, features*4, down=True, act="relu", use_dropout=True)
        self.down3 = Block(features*4, features*8, down=True, act="relu", use_dropout=True)
        self.down4 = Block(features*8, features*8, down=True, act="relu", use_dropout=True)
        self.down5 = Block(features*8, features*8, down=True, act="relu", use_dropout=True)
        self.down6 = Block(features*8, features*8, down=True, act="relu", use_dropout=True)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(features*8, features*8, 4, padding='same'),
            nn.ReLU()
        )
        self.bottleneck_up = nn.Sequential(
            nn.ConvTranspose2d(features*8, features*8, 5, 2, 1),
            nn.ReLU()
        )
        
        # Decoder
        self.up1 = Block(features*8, features*8*2, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features*8, features*4, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features*4, features, down=False, act="relu", use_dropout=False)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features, out_channels, kernel_size=(3,5), stride=(1,3), padding=(1,1)),
        )
    
    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        
        up1 = self.up1(d5)
        up2 = self.up2(up1)
        up3 = self.up3(up2)
        up4 = self.up4(up3)
        
        return self.final_up(up4)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=True, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=True),
            
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act=="relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


# ============================================================================
# STRATEGY 4: Climatology-Based Regularization
# ============================================================================

class ClimatologyLoss(nn.Module):
    """
    Penalize predictions that deviate too much from climatology.
    """
    def __init__(self, climatology_data, lambda_clim=0.1):
        """
        Args:
            climatology_data: numpy array of shape (12, 48, 144) - monthly climatology
            lambda_clim: weight for climatology loss
        """
        super().__init__()
        self.climatology = torch.FloatTensor(climatology_data).to(device)  # (12, 48, 144)
        self.lambda_clim = lambda_clim
        
    def forward(self, predicted_sst, target_months):
        """
        Args:
            predicted_sst: tensor of shape (batch, 1, 48, 144)
            target_months: tensor of shape (batch,) with month indices
        
        Returns:
            climatology loss value
        """
        batch_size = predicted_sst.shape[0]
        clim_loss = 0.0
        
        for i in range(batch_size):
            month = int(target_months[i].item())
            expected_clim = self.climatology[month].unsqueeze(0)  # (1, 48, 144)
            
            # Calculate anomaly from climatology
            pred_anomaly = predicted_sst[i] - expected_clim
            
            # Penalize large anomalies
            clim_loss += torch.mean(pred_anomaly ** 2)
        
        return self.lambda_clim * clim_loss / batch_size


# ============================================================================
# Combined Loss Function
# ============================================================================

class CombinedLoss(nn.Module):
    """
    Combines reconstruction loss with climatology regularization.
    """
    def __init__(self, climatology_data=None, lambda_clim=0.1, lambda_l1=100.0):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.lambda_l1 = lambda_l1
        
        if climatology_data is not None:
            self.clim_loss = ClimatologyLoss(climatology_data, lambda_clim)
            self.use_clim_loss = True
        else:
            self.use_clim_loss = False
    
    def forward(self, predicted, target, target_months=None):
        # Reconstruction loss
        l1 = self.l1_loss(predicted, target)
        
        # Climatology loss
        clim = 0.0
        if self.use_clim_loss and target_months is not None:
            clim = self.clim_loss(predicted, target_months)
        
        total_loss = self.lambda_l1 * l1 + clim
        
        return total_loss, {'l1': l1.item(), 'climatology': clim if isinstance(clim, float) else clim.item()}


# ============================================================================
# Training Function
# ============================================================================

def train_epoch(generator, discriminator, train_loader, g_optimizer, d_optimizer, 
                criterion, bce_loss, epoch):
    """
    Train for one epoch.
    """
    generator.train()
    discriminator.train()
    
    total_g_loss = 0
    total_d_loss = 0
    total_l1_loss = 0
    total_clim_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        target_months = batch['target_month'].to(device)
        
        batch_size = inputs.shape[0]
        
        # ===========================
        # Train Discriminator
        # ===========================
        d_optimizer.zero_grad()
        
        # Real pairs
        real_pair = torch.cat([inputs, targets], dim=1)
        d_real = discriminator(real_pair)
        real_labels = torch.ones_like(d_real)
        d_real_loss = bce_loss(d_real, real_labels)
        
        # Fake pairs
        fake_outputs = generator(inputs)
        fake_pair = torch.cat([inputs, fake_outputs.detach()], dim=1)
        d_fake = discriminator(fake_pair)
        fake_labels = torch.zeros_like(d_fake)
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
        
        # Adversarial loss
        g_adv_loss = bce_loss(d_fake, real_labels)
        
        # Reconstruction + Climatology loss
        g_recon_loss, loss_dict = criterion(fake_outputs, targets, target_months)
        
        # Total generator loss
        g_loss = g_adv_loss + g_recon_loss
        g_loss.backward()
        g_optimizer.step()
        
        # Accumulate losses
        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
        total_l1_loss += loss_dict['l1']
        total_clim_loss += loss_dict['climatology']
        
        if batch_idx % 50 == 0:
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'G_loss: {g_loss.item():.4f} (L1: {loss_dict["l1"]:.4f}, '
                  f'Clim: {loss_dict["climatology"]:.4f}) D_loss: {d_loss.item():.4f}')
    
    return {
        'g_loss': total_g_loss / len(train_loader),
        'd_loss': total_d_loss / len(train_loader),
        'l1_loss': total_l1_loss / len(train_loader),
        'clim_loss': total_clim_loss / len(train_loader)
    }


# ============================================================================
# Rollout Prediction Function
# ============================================================================

def rollout_prediction(generator, initial_inputs, initial_month, num_months=12):
    """
    Perform rollout prediction for multiple months.
    
    Args:
        generator: trained generator model
        initial_inputs: tensor of shape (num_input_months, 48, 144) - initial SST fields
        initial_month: int (0-11) - the month corresponding to the LAST input
        num_months: int - number of months to predict forward
    
    Returns:
        predictions: list of numpy arrays (48, 144)
    """
    generator.eval()
    predictions = []
    
    # Convert to list for easier manipulation
    current_inputs = [initial_inputs[i].unsqueeze(0).to(device) for i in range(initial_inputs.shape[0])]
    current_month = initial_month
    
    with torch.no_grad():
        for step in range(num_months):
            # Stack current inputs
            input_sst = torch.cat(current_inputs, dim=0).unsqueeze(0).to(device)  # (1, num_input_months, 48, 144)
            
            # Get seasonal encoding for current month
            seasonal_encoding = get_seasonal_encoding(current_month).unsqueeze(0).to(device)  # (1, 2, 48, 144)
            
            # Combine inputs
            model_input = torch.cat([input_sst.squeeze(0), seasonal_encoding.squeeze(0)], dim=0).unsqueeze(0).to(device)  # (1, num_input_months + 2, 48, 144)
            
            # Predict next month
            prediction = generator(model_input)  # (1, 1, 48, 144)
            
            # Store prediction
            pred_np = prediction.squeeze().cpu().numpy()
            predictions.append(pred_np)
            
            # Update inputs for next iteration
            current_inputs.pop(0)  # Remove oldest month
            current_inputs.append(prediction.squeeze(0))  # Add new prediction
            
            # Update month
            current_month = (current_month + 1) % 12
    
    return predictions

def rollout_prediction_new(generator, initial_inputs, initial_month, num_months=12):
    """
    Perform rollout prediction for multiple months.
    
    Args:
        generator: trained generator model
        initial_inputs: tensor of shape (num_input_months, 48, 144) - initial SST fields
        initial_month: int (0-11) - the month corresponding to the LAST input
        num_months: int - number of months to predict forward
    
    Returns:
        predictions: list of numpy arrays (48, 144)
    """
    generator.eval()
    predictions = []
    
    # Convert to list for easier manipulation
    current_inputs = [initial_inputs[i].unsqueeze(0).to(device) for i in range(initial_inputs.shape[0])]
    current_month = initial_month
    
    with torch.no_grad():
        for step in range(num_months):
            # Stack current inputs
            input_sst = torch.cat(current_inputs, dim=0).unsqueeze(0).to(device)  # (1, num_input_months, 48, 144)
            
            # Get seasonal encoding for current month
            seasonal_encoding = get_seasonal_encoding(current_month).unsqueeze(0).to(device)  # (1, 2, 48, 144)
            
            # Combine inputs
            model_input = torch.cat([input_sst.squeeze(0), seasonal_encoding.squeeze(0)], dim=0).unsqueeze(0).to(device)  # (1, num_input_months + 2, 48, 144)
            
            # Predict next month
            prediction = generator(input_sst.to(device))  # (1, 1, 48, 144)
            
            # Store prediction
            pred_np = prediction.squeeze().cpu().numpy()
            predictions.append(pred_np)
            
            # Update inputs for next iteration
            current_inputs.pop(0)  # Remove oldest month
            current_inputs.append(prediction.squeeze(0))  # Add new prediction
            
            # Update month
            current_month = (current_month + 1) % 12
    
    return predictions

# ============================================================================
# Example Usage
# ============================================================================

def compute_climatology(sst_data, months):
    """
    Compute monthly climatology from data.
    
    Args:
        sst_data: numpy array of shape (num_samples, 48, 144)
        months: numpy array of shape (num_samples,) with month indices
    
    Returns:
        climatology: numpy array of shape (12, 48, 144)
    """
    climatology = np.zeros((12, 48, 144))
    
    for month in range(12):
        month_mask = months == month
        climatology[month] = np.mean(sst_data[month_mask], axis=0)
    
    return climatology


if __name__ == "__main__":
    # Example setup
    num_input_months = 3  # Use 3 previous months as input
    in_channels = num_input_months + 2  # SST months + 2 seasonal encoding channels
    
    # Load your data (placeholder)
    # sst_data: shape (num_samples, 48, 144)
    # months: shape (num_samples,) with values 0-11
    
    # Example with dummy data
    num_samples = 1200  # 100 years of monthly data
    sst_data = np.random.randn(num_samples, 48, 144).astype(np.float32)
    months = np.array([i % 12 for i in range(num_samples)])
    
    # Compute climatology
    climatology = compute_climatology(sst_data, months)
    
    # Create dataset
    train_dataset = SSTDataset(
        sst_data=sst_data,
        months=months,
        num_input_months=num_input_months,
        climatology=climatology
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    
    # Initialize models
    generator = Generator(in_channels=in_channels, features=128, out_channels=1).to(device)
    
    # Discriminator (you'll need to define this)
    # discriminator = Discriminator(in_channels=in_channels+1).to(device)
    
    # Loss functions
    criterion = CombinedLoss(climatology_data=climatology, lambda_clim=0.1, lambda_l1=100.0)
    bce_loss = nn.BCEWithLogitsLoss()
    
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Input channels: {in_channels}")
    print(f"Number of input months: {num_input_months}")
    print("Ready to train!")
    
    # Example rollout
    # initial_sst = torch.randn(num_input_months, 48, 144)
    # predictions = rollout_prediction(generator, initial_sst, initial_month=0, num_months=12)
    # print(f"Generated {len(predictions)} monthly predictions")
