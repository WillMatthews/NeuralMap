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
from src.run_utils import generate_run_id, save_run_metadata


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


def train_zoom_level(model, train_loader, val_loader, criterion, optimizer, device, 
                     writer, zoom_level, num_epochs, save_interval, log_interval,
                     models_dir, global_epoch_offset, run_id):
    """Train model on a specific zoom level."""
    best_val_loss = float('inf')
    
    print(f"\n{'='*60}")
    print(f"Training on zoom levels 0-{zoom_level}")
    print(f"Epochs: {num_epochs}")
    print(f"{'='*60}\n")
    
    for epoch in range(1, num_epochs + 1):
        global_epoch = global_epoch_offset + epoch
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            device, global_epoch, writer, log_interval
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, global_epoch, writer)
        
        epoch_time = time.time() - start_time
        print(f"Zoom 0-{zoom_level} | Epoch {epoch}/{num_epochs} (Global: {global_epoch}) - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} "
              f"({epoch_time:.1f}s)")
        
        # Save checkpoint
        if epoch % save_interval == 0 or val_loss < best_val_loss:
            checkpoint = {
                'epoch': global_epoch,
                'zoom_level': zoom_level,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save with run ID prefix
                torch.save(checkpoint, models_dir / f'{run_id}_best_model.pt')
                # Also save as best_model.pt for backward compatibility
                torch.save(checkpoint, models_dir / 'best_model.pt')
                print(f"Saved best model (val_loss: {val_loss:.4f})")
            
            torch.save(checkpoint, models_dir / f'{run_id}_checkpoint_epoch_{global_epoch}.pt')
    
    return global_epoch


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate run ID and save metadata
    run_id = generate_run_id()
    runs_dir = Path(config['paths'].get('runs_dir', 'runs'))
    models_dir = Path(config['paths']['models_dir'])
    logs_dir = Path(config['paths']['logs_dir'])
    
    # Create directories
    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create run-specific directories
    run_logs_dir = logs_dir / run_id
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}\n")
    
    # Save run metadata
    metadata_file = save_run_metadata(
        run_id=run_id,
        config=config,
        runs_dir=runs_dir,
        additional_metadata={
            'device': str(device),
            'logs_dir': str(run_logs_dir),
            'models_dir': str(models_dir)
        }
    )
    print(f"Run metadata saved to: {metadata_file}")
    
    # Model
    model_config = config['model']
    data_config = config['data']
    
    # Use new neural field architecture
    from src.model import CoordinateNeuralField
    model = CoordinateNeuralField(
        hidden_dim=model_config['hidden_dim'],
        num_mlp_layers=model_config.get('num_mlp_layers', model_config.get('num_layers', 8)),
        num_frequencies=model_config.get('num_frequencies', 10),
        tile_size=data_config['tile_size'],
        dropout=model_config['dropout'],
        num_attention_blocks=model_config.get('num_attention_blocks', 2)
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    # TensorBoard - use run-specific log directory
    writer = SummaryWriter(log_dir=str(run_logs_dir))
    
    # Training configuration
    save_interval = config['training']['save_interval']
    log_interval = config['training']['log_interval']
    min_zoom = data_config['min_zoom']
    max_zoom = data_config['max_zoom']
    
    # Check if hierarchical training is enabled
    hierarchical = config['training'].get('hierarchical_training', False)
    
    if hierarchical:
        base_epochs = config['training'].get('base_epochs_per_zoom', 20)
        print(f"\n{'='*60}")
        print("HIERARCHICAL TRAINING MODE")
        print(f"Training progressively: zoom 0 -> zoom {max_zoom}")
        print(f"Base epochs per zoom: {base_epochs}")
        print(f"{'='*60}\n")
        
        global_epoch_offset = 0
        
        # Train progressively on each zoom level
        # Each zoom level has 4x the information of the next level,
        # so we train 4x more epochs for lower zoom levels
        # This ensures the model learns global patterns (zoom 0) perfectly
        # before adding finer details at higher zoom levels
        print("\nTraining schedule:")
        total_epochs = 0
        for z in range(min_zoom, max_zoom + 1):
            zoom_diff = max_zoom - z
            epochs = base_epochs * (4 ** zoom_diff)
            total_epochs += epochs
            print(f"  Zoom 0-{z}: {epochs:,} epochs")
        print(f"Total epochs: {total_epochs:,}\n")
        
        for current_max_zoom in range(min_zoom, max_zoom + 1):
            zoom_diff = max_zoom - current_max_zoom
            num_epochs = base_epochs * (4 ** zoom_diff)
            
            print(f"\nPreparing datasets for zoom levels 0-{current_max_zoom}...")
            
            # Create datasets filtered by current max zoom
            train_dataset = MapTileDataset(
                config_path='config.yaml', 
                split='train',
                max_zoom_filter=current_max_zoom
            )
            val_dataset = MapTileDataset(
                config_path='config.yaml', 
                split='val',
                max_zoom_filter=current_max_zoom
            )
            
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
            
            # Train on this zoom level
            global_epoch_offset = train_zoom_level(
                model, train_loader, val_loader, criterion, optimizer,
                device, writer, current_max_zoom, num_epochs,
                save_interval, log_interval, models_dir, global_epoch_offset, run_id
            )
    else:
        # Standard training: all zoom levels at once
        print("\nStandard training mode: all zoom levels together\n")
        
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
        
        num_epochs = config['training']['num_epochs']
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
                    # Save with run ID prefix
                    torch.save(checkpoint, models_dir / f'{run_id}_best_model.pt')
                    # Also save as best_model.pt for backward compatibility
                    torch.save(checkpoint, models_dir / 'best_model.pt')
                    print(f"Saved best model (val_loss: {val_loss:.4f})")
                
                torch.save(checkpoint, models_dir / f'{run_id}_checkpoint_epoch_{epoch}.pt')
    
    writer.close()
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
