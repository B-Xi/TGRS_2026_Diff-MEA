import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# Diffusion model components.
from unet3d import SimpleUnet
from diffusion import Diffusion
from utils import AvgrageMeter

# Training device.
device = torch.device('cuda:1')

# Define paths
feature_path = "_features_Houston.npy"
save_dir = "."

# Hyperparameters
batch_size = 64
epochs = 1000000
save_interval = 100  # Save a checkpoint every 100 epochs.
lr = 1e-4
T = 600 


class FeatureDataset(Dataset):
    def __init__(self, feature_path):
        """
        Dataset for loading feature tensors
        
        Args:
            feature_path: Path to the .npy file containing features
        """
        # Load feature data - shape [2520, 9, 9, 64]
        self.features = np.load(feature_path)
        print(f"Loaded features with shape: {self.features.shape}")
        
        # Convert to [batch, channel, depth, height, width] format for 3D UNet
        # Current format: [N, H, W, C] → Target format: [N, 1, C, H, W]
        self.features = np.transpose(self.features, (0, 1, 2, 3))
        self.features = np.expand_dims(self.features, 1)
        print(f"Reshaped features to: {self.features.shape}")

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        # Convert to float32 tensor
        feature_tensor = torch.from_numpy(feature.astype(np.float32))
        return feature_tensor, torch.zeros(1)  # Dummy label


def save_model(model, epoch):
    """Save model checkpoint"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    path = f"{save_dir}/feature_diffusion_epoch_{epoch}.pt"
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def train():
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # Load dataset
    print("Loading feature dataset...")
    dataset = FeatureDataset(feature_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize diffusion model
    print(f"Initializing diffusion model with T={T}...")
    diffusion = Diffusion(T=T)
    
    # For features with shape [batch, 1, 64, 9, 9], we need to set channels=64
    feature_channels = dataset.features.shape[2]  # Should be 64
    print(f"Creating UNet3D with feature channels={feature_channels}")
    model = SimpleUnet(_image_channels=1)  # 1 channel (the feature volumes are single channel)
    device = torch.device('cuda:1')
    model.to(device)
    
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    loss_metric = AvgrageMeter()
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        loss_metric.reset()
        
        for step, (batch, _) in enumerate(dataloader):
            batch = batch.to(device)
            batch = batch.permute(0,1,4,2,3)
            optimizer.zero_grad()
            
            # Sample random timesteps
            cur_batch_size = batch.shape[0]
            t = torch.randint(0, diffusion.T, (cur_batch_size,), device=device).long()
            
            # Calculate loss
            loss, temp_xt, temp_noise, temp_noise_pred = diffusion.get_loss(model, batch, t)
            loss.backward()
            optimizer.step()
            
            loss_metric.update(loss.item(), batch.shape[0])
            
            if step % 10 == 0:
                print(f"[Epoch {epoch}, Step {step:03d}] Loss: {loss.item():.6f}")
        
        # Print epoch summary
        print(f"[TRAIN EPOCH {epoch}] Loss: {loss_metric.get_avg():.6f}")
        
        # Save model checkpoint at specified intervals
        if epoch % save_interval == 0 or epoch == epochs - 1:
            save_model(model, epoch)


if __name__ == "__main__":
    train()
