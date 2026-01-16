"""Download OSM tiles for training."""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import random
import yaml
from pathlib import Path
from tqdm import tqdm
import time

from src.data_utils import (
    deg2num, get_tiles_in_region, download_tile,
    get_tile_center_lat_lon
)


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    min_lat = config['region']['min_lat']
    max_lat = config['region']['max_lat']
    min_lon = config['region']['min_lon']
    max_lon = config['region']['max_lon']
    
    min_zoom = config['data']['min_zoom']
    max_zoom = config['data']['max_zoom']
    num_tiles_per_zoom = config['data']['num_tiles_per_zoom']
    cache_dir = config['data']['cache_dir']
    tiles_dir = config['data']['tiles_dir']
    
    # Create directories
    Path(tiles_dir).mkdir(parents=True, exist_ok=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading tiles for region: ({min_lat}, {min_lon}) to ({max_lat}, {max_lon})")
    print(f"Zoom levels: {min_zoom} to {max_zoom}")
    print(f"Target: {num_tiles_per_zoom} tiles per zoom level")
    
    total_tiles = 0
    
    # Check if we're doing full globe
    is_full_globe = (min_lat <= -90 and max_lat >= 90 and 
                     min_lon <= -180 and max_lon >= 180)
    
    for zoom in range(min_zoom, max_zoom + 1):
        print(f"\nProcessing zoom level {zoom}...")
        
        # For full globe, generate random tile coordinates directly (more efficient)
        if is_full_globe:
            max_tile = 2 ** zoom
            # Generate random tile coordinates
            tiles = []
            for _ in range(num_tiles_per_zoom):
                x = random.randint(0, max_tile - 1)
                y = random.randint(0, max_tile - 1)
                tiles.append((x, y))
            print(f"Generated {len(tiles)} random tiles for zoom {zoom} (full globe)")
        else:
            # Get all tiles in region
            all_tiles = get_tiles_in_region(min_lat, max_lat, min_lon, max_lon, zoom)
            
            # Sample random tiles if we have more than needed
            if len(all_tiles) > num_tiles_per_zoom:
                tiles = random.sample(all_tiles, num_tiles_per_zoom)
            else:
                tiles = all_tiles
            print(f"Downloading {len(tiles)} tiles for zoom {zoom} (from {len(all_tiles)} in region)")
        
        # Create zoom directory
        zoom_dir = Path(tiles_dir) / str(zoom)
        zoom_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded = 0
        for xtile, ytile in tqdm(tiles, desc=f"Zoom {zoom}"):
            # Download tile
            cache_path = download_tile(xtile, ytile, zoom, cache_dir)
            
            if cache_path and cache_path.exists():
                # Copy to tiles directory
                tile_path = zoom_dir / f"{zoom}_{xtile}_{ytile}.png"
                if not tile_path.exists():
                    import shutil
                    shutil.copy(cache_path, tile_path)
                downloaded += 1
                total_tiles += 1
            
            # Be nice to OSM servers
            time.sleep(0.1)
        
        print(f"Downloaded {downloaded}/{len(tiles)} tiles for zoom {zoom}")
    
    print(f"\nTotal tiles downloaded: {total_tiles}")
    print(f"Tiles saved to: {tiles_dir}")


if __name__ == '__main__':
    main()
