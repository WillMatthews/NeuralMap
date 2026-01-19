"""State-of-the-art coordinate-based neural field model for map compression."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FourierFeatureEncoding(nn.Module):
    """
    Fourier feature encoding for coordinates (NeRF-style).
    Encodes coordinates using sinusoidal functions at multiple frequencies,
    enabling the network to learn high-frequency details.
    """
    
    def __init__(self, input_dim=3, num_frequencies=10, include_input=True):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        
        # Create frequency bands (log-linear spacing)
        # Frequencies: 2^0, 2^1, ..., 2^(num_frequencies-1)
        self.register_buffer('freq_bands', 
                            2.0 ** torch.arange(0, num_frequencies, dtype=torch.float32))
        
        # Output dimension: input_dim * (2 * num_frequencies) + (input_dim if include_input)
        self.output_dim = input_dim * (2 * num_frequencies) + (input_dim if include_input else 0)
    
    def forward(self, coords):
        """
        Args:
            coords: (batch_size, input_dim) tensor of coordinates
        Returns:
            encoded: (batch_size, output_dim) tensor of encoded coordinates
        """
        # Expand coordinates: (batch, input_dim) -> (batch, input_dim, num_freqs)
        coords_expanded = coords.unsqueeze(-1)  # (batch, input_dim, 1)
        freqs = self.freq_bands.view(1, 1, -1)  # (1, 1, num_freqs)
        
        # Compute sin and cos for each frequency
        # (batch, input_dim, num_freqs)
        coords_scaled = coords_expanded * freqs
        sin_coords = torch.sin(coords_scaled)
        cos_coords = torch.cos(coords_scaled)
        
        # Interleave sin and cos: (batch, input_dim, 2*num_freqs)
        encoded = torch.stack([sin_coords, cos_coords], dim=-1)
        encoded = encoded.view(coords.size(0), self.input_dim, 2 * self.num_frequencies)
        encoded = encoded.view(coords.size(0), -1)  # (batch, input_dim * 2 * num_freqs)
        
        # Optionally include original coordinates
        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)
        
        return encoded


class ResidualMLPBlock(nn.Module):
    """Residual MLP block with layer normalization and skip connections."""
    
    def __init__(self, dim, activation=nn.ReLU(), dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + residual


class CoordinateMLP(nn.Module):
    """
    Coordinate-based MLP that processes encoded coordinates.
    Uses residual connections and layer normalization for better training.
    """
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual MLP blocks
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, activation=nn.ReLU(), dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, encoded_coords):
        """
        Args:
            encoded_coords: (batch_size, input_dim) encoded coordinates
        Returns:
            features: (batch_size, hidden_dim) feature vector
        """
        x = self.input_proj(encoded_coords)
        
        # Pass through residual blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = self.output_proj(x)
        
        return x


class AttentionBlock(nn.Module):
    """Self-attention block for spatial features."""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, height*width, dim) or (batch, dim, height, width)
        Returns:
            out: same shape as input
        """
        # Handle both formats
        if x.dim() == 4:
            batch, dim, h, w = x.shape
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(batch, h * w, dim)
            reshape_back = True
        else:
            reshape_back = False
        
        residual = x
        x = self.norm(x)
        attn_out, _ = self.attention(x, x, x)
        x = residual + self.dropout(attn_out)
        
        if reshape_back:
            batch, hw, dim = x.shape
            h = w = int(math.sqrt(hw))
            x = x.view(batch, h, w, dim)
            x = x.permute(0, 3, 1, 2).contiguous()
        
        return x


class UNetDecoder(nn.Module):
    """
    U-Net style decoder with attention mechanisms.
    Generates high-resolution images from coordinate features.
    """
    
    def __init__(self, input_dim=256, tile_size=256, num_attention_blocks=2):
        super().__init__()
        self.tile_size = tile_size
        self.input_dim = input_dim
        
        # Determine number of upsampling stages
        # Start from a small feature map (e.g., 4x4 or 8x8)
        initial_size = 4
        num_stages = int(math.log2(tile_size // initial_size))
        
        # Initial feature map projection
        self.initial_proj = nn.Linear(input_dim, initial_size * initial_size * 64)
        self.initial_size = initial_size
        
        # Build upsampling path
        channels = [64, 128, 256, 512, 256, 128, 64, 32]
        # Adjust channels based on number of stages
        if num_stages < len(channels):
            channels = channels[:num_stages+1]
        else:
            channels = channels + [channels[-1]] * (num_stages - len(channels) + 1)
        
        self.upsample_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        for i in range(num_stages):
            in_ch = channels[min(i, len(channels)-1)]
            out_ch = channels[min(i+1, len(channels)-1)] if i+1 < len(channels) else 3
            
            # Upsampling block
            self.upsample_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, out_ch) if out_ch >= 8 else nn.BatchNorm2d(out_ch),
                nn.SiLU() if i < num_stages - 1 else nn.Sigmoid()  # Sigmoid for final output
            ))
            
            # Attention block (skip last stage)
            if i < num_stages - 1 and i < num_attention_blocks:
                self.attention_blocks.append(AttentionBlock(out_ch, num_heads=8))
            else:
                self.attention_blocks.append(None)
    
    def forward(self, features):
        """
        Args:
            features: (batch_size, input_dim) coordinate features
        Returns:
            images: (batch_size, 3, tile_size, tile_size) RGB images
        """
        batch_size = features.size(0)
        
        # Project to initial spatial features
        spatial = self.initial_proj(features)
        h = w = self.initial_size
        x = spatial.view(batch_size, 64, h, w)
        
        # Upsample through stages
        for i, (upsample, attn) in enumerate(zip(self.upsample_blocks, self.attention_blocks)):
            x = upsample(x)
            
            # Apply attention if available
            if attn is not None:
                x = attn(x)
        
        # Ensure correct output size
        if x.size(2) != self.tile_size or x.size(3) != self.tile_size:
            x = F.interpolate(
                x, size=(self.tile_size, self.tile_size),
                mode='bilinear', align_corners=False
            )
        
        return x


class CoordinateNeuralField(nn.Module):
    """
    State-of-the-art coordinate-based neural field for map tile generation.
    
    Architecture:
    1. Fourier feature encoding of coordinates (lat, lon, zoom)
    2. Coordinate-based MLP with residual connections
    3. U-Net decoder with attention for high-quality image generation
    """
    
    def __init__(self, hidden_dim=256, num_mlp_layers=8, num_frequencies=10,
                 tile_size=256, dropout=0.1, num_attention_blocks=2):
        super().__init__()
        self.tile_size = tile_size
        self.hidden_dim = hidden_dim
        
        # Fourier feature encoding
        self.coord_encoder = FourierFeatureEncoding(
            input_dim=3,  # lat, lon, zoom
            num_frequencies=num_frequencies,
            include_input=True
        )
        
        # Coordinate-based MLP
        self.coord_mlp = CoordinateMLP(
            input_dim=self.coord_encoder.output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            dropout=dropout
        )
        
        # U-Net decoder
        self.decoder = UNetDecoder(
            input_dim=hidden_dim,
            tile_size=tile_size,
            num_attention_blocks=num_attention_blocks
        )
    
    def forward(self, coords):
        """
        Args:
            coords: (batch_size, 3) tensor of [lat, lon, zoom] (normalized)
        Returns:
            images: (batch_size, 3, tile_size, tile_size) RGB images
        """
        # Encode coordinates with Fourier features
        encoded_coords = self.coord_encoder(coords)
        
        # Process through coordinate MLP
        features = self.coord_mlp(encoded_coords)
        
        # Decode to image
        images = self.decoder(features)
        
        return images


# Backward compatibility alias
CoordinateTransformer = CoordinateNeuralField
