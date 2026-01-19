"""Dataset class for map tiles."""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import yaml

from src.data_utils import (
    load_tile_image, get_tile_center_lat_lon, 
    lat_lon_to_normalized, deg2num, get_tiles_in_region
)


class MapTileDataset(Dataset):
    """Dataset for map tiles with coordinate inputs."""
    
    def __init__(self, config_path='config.yaml', split='train', max_zoom_filter=None):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.tile_size = config['data']['tile_size']
        self.min_zoom = config['data']['min_zoom']
        self.max_zoom = config['data']['max_zoom']
        self.cache_dir = config['data']['cache_dir']
        self.tiles_dir = config['data']['tiles_dir']
        
        # Region bounds
        self.min_lat = config['region']['min_lat']
        self.max_lat = config['region']['max_lat']
        self.min_lon = config['region']['min_lon']
        self.max_lon = config['region']['max_lon']
        
        # Load tile list
        self.tiles = self._load_tile_list()
        
        # Filter by max zoom if specified (for hierarchical training)
        if max_zoom_filter is not None:
            self.tiles = [(z, x, y) for z, x, y in self.tiles if z <= max_zoom_filter]
        
        # Split dataset (if split is None or 'all', use all tiles)
        if split is None or split == 'all':
            # Use all tiles - no splitting for learning the entire globe
            pass
        else:
            # Only split if explicitly requested
            random.seed(42)
            random.shuffle(self.tiles)
            split_idx = int(len(self.tiles) * 0.9)
            # Ensure at least 1 sample in each split for very small datasets
            if split_idx == 0 and len(self.tiles) > 0:
                split_idx = min(1, len(self.tiles))
            elif split_idx == len(self.tiles) and len(self.tiles) > 1:
                split_idx = len(self.tiles) - 1
            if split == 'train':
                self.tiles = self.tiles[:split_idx]
            elif split == 'val':
                self.tiles = self.tiles[split_idx:]
    
    def _load_tile_list(self):
        """Load list of available tiles."""
        tiles = []
        tiles_dir = Path(self.tiles_dir)
        
        if not tiles_dir.exists():
            return tiles
        
        # Look for tile files or metadata
        for zoom in range(self.min_zoom, self.max_zoom + 1):
            zoom_dir = tiles_dir / str(zoom)
            if zoom_dir.exists():
                for tile_file in zoom_dir.glob('*.png'):
                    # Parse filename: zoom_xtile_ytile.png
                    parts = tile_file.stem.split('_')
                    if len(parts) == 3:
                        xtile, ytile = int(parts[1]), int(parts[2])
                        tiles.append((zoom, xtile, ytile))
        
        return tiles
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        zoom, xtile, ytile = self.tiles[idx]
        
        # Get tile center coordinates
        lat, lon = get_tile_center_lat_lon(xtile, ytile, zoom)
        
        # Normalize coordinates
        coords = lat_lon_to_normalized(
            lat, lon, zoom,
            self.min_lat, self.max_lat,
            self.min_lon, self.max_lon
        )
        
        # Load tile image
        tile_path = Path(self.tiles_dir) / str(zoom) / f"{zoom}_{xtile}_{ytile}.png"
        if not tile_path.exists():
            # Try cache directory
            cache_path = Path(self.cache_dir) / f"{zoom}_{xtile}_{ytile}.png"
            tile_path = cache_path if cache_path.exists() else None
        
        image = load_tile_image(tile_path, self.tile_size)
        
        if image is None:
            # Return black image if tile not found
            image = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.float32)
        
        # Convert to tensor and change to CHW format
        image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
        
        return {
            'coords': torch.from_numpy(coords).float(),
            'image': image,
            'zoom': zoom,
            'tile': (xtile, ytile)
        }
