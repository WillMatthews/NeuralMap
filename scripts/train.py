"""Training script for map compression model."""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from pathlib import Path
from tqdm import tqdm
import time

from src.model import CoordinateTransformer
from src.dataset import MapTileDataset


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, log_interval):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        coords = batch['coords'].to(device)
        images = batch['image'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_images = model(coords)
        
        # Loss
        loss = criterion(pred_images, images)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Logging
        if batch_idx % log_interval == 0:
            writer.add_scalar('Train/Loss', loss.item(), 
                            epoch * len(dataloader) + batch_idx)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, criterion, device, epoch, writer):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            coords = batch['coords'].to(device)
            images = batch['image'].to(device)
            
            pred_images = model(coords)
            loss = criterion(pred_images, images)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    
    # Log sample images
    if num_batches > 0:
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(dataloader))
            coords = sample_batch['coords'][:4].to(device)
            images = sample_batch['image'][:4].to(device)
            pred_images = model(coords)
            
            # Log images to tensorboard
            writer.add_images('Val/GT', images, epoch)
            writer.add_images('Val/Pred', pred_images, epoch)
    
    return avg_loss


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    models_dir = Path(config['paths']['models_dir'])
    logs_dir = Path(config['paths']['logs_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset
    train_dataset = MapTileDataset(config_path='config.yaml', split='train')
    val_dataset = MapTileDataset(config_path='config.yaml', split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Model
    model_config = config['model']
    data_config = config['data']
    model = CoordinateTransformer(
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        positional_encoding_dim=model_config['positional_encoding_dim'],
        tile_size=data_config['tile_size']
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(logs_dir))
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    save_interval = config['training']['save_interval']
    log_interval = config['training']['log_interval']
    
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, writer, log_interval
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, epoch, writer)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} "
              f"({epoch_time:.1f}s)")
        
        # Save checkpoint
        if epoch % save_interval == 0 or val_loss < best_val_loss:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, models_dir / 'best_model.pt')
                print(f"Saved best model (val_loss: {val_loss:.4f})")
            
            torch.save(checkpoint, models_dir / f'checkpoint_epoch_{epoch}.pt')
    
    writer.close()
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
