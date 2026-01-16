"""Coordinate-based transformer model for map compression."""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for coordinates."""
    
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


class CoordinateTransformer(nn.Module):
    """Coordinate-based transformer that maps lat/lon/zoom to RGB tiles."""
    
    def __init__(self, hidden_dim=256, num_layers=8, num_heads=8, 
                 dropout=0.1, positional_encoding_dim=128, tile_size=256):
        super().__init__()
        self.tile_size = tile_size
        self.hidden_dim = hidden_dim
        
        # Input: lat, lon, zoom (3D coordinates)
        # Use positional encoding for coordinates
        self.input_proj = nn.Linear(3, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder: hidden features -> RGB image
        # Use a CNN decoder for spatial structure
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.ReLU(),
            nn.Linear(hidden_dim * 8, hidden_dim * 16),
            nn.ReLU(),
        )
        
        # Spatial decoder: reshape to image dimensions
        # We'll generate a feature map then convert to RGB
        # For 256x256: 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256 (5 upsamples)
        # For 512x512: 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256 -> 512x512 (6 upsamples)
        spatial_size = 8  # Start with 8x8 feature map
        self.spatial_proj = nn.Linear(hidden_dim * 16, spatial_size * spatial_size * 64)
        
        # CNN upsampling to final image size
        if tile_size == 256:
            # 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 8->16
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 16->32
                nn.ReLU(),
                nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),   # 32->64
                nn.ReLU(),
                nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),   # 64->128
                nn.ReLU(),
                nn.ConvTranspose2d(4, 3, kernel_size=4, stride=2, padding=1),    # 128->256
                nn.Sigmoid()  # Output in [0, 1] range
            )
        else:  # 512
            # 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256 -> 512x512
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 8->16
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 16->32
                nn.ReLU(),
                nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),   # 32->64
                nn.ReLU(),
                nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),    # 64->128
                nn.ReLU(),
                nn.ConvTranspose2d(4, 3, kernel_size=4, stride=2, padding=1),   # 128->256
                nn.ReLU(),
                nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),    # 256->512
                nn.Sigmoid()
            )
    
    def forward(self, coords):
        """
        Args:
            coords: (batch_size, 3) tensor of [lat, lon, zoom] (normalized)
        Returns:
            images: (batch_size, 3, tile_size, tile_size) RGB images
        """
        batch_size = coords.size(0)
        
        # Project coordinates to hidden dimension
        # Add sequence dimension for transformer (treat as sequence of length 1)
        x = self.input_proj(coords).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Apply positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, 1, hidden_dim)
        x = x.squeeze(1)  # (batch, hidden_dim)
        
        # Decode to features
        features = self.decoder(x)  # (batch, hidden_dim * 16)
        
        # Project to spatial features
        spatial_features = self.spatial_proj(features)
        
        # Reshape for CNN (always start at 8x8)
        h, w = 8, 8
        spatial_features = spatial_features.view(batch_size, 64, h, w)
        
        # Upsample to final image size
        images = self.upsample(spatial_features)
        
        # Ensure correct output size
        if images.size(2) != self.tile_size or images.size(3) != self.tile_size:
            images = nn.functional.interpolate(
                images, size=(self.tile_size, self.tile_size), 
                mode='bilinear', align_corners=False
            )
        
        return images
