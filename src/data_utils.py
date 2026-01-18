"""Utilities for downloading and processing OSM tiles."""
import os
import math
import random
import requests
from PIL import Image
import numpy as np
from pathlib import Path
import yaml


def deg2num(lat_deg, lon_deg, zoom):
    """Convert lat/lon to tile coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    max_tile = n - 1
    
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    
    # Clamp to valid tile coordinate ranges [0, 2^zoom - 1]
    xtile = max(0, min(max_tile, xtile))
    ytile = max(0, min(max_tile, ytile))
    
    return (xtile, ytile)


def num2deg(xtile, ytile, zoom):
    """Convert tile coordinates to lat/lon of top-left corner."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def get_tile_url(xtile, ytile, zoom, style='default'):
    """Get OSM tile URL."""
    # Using OpenStreetMap tile server
    # Note: In production, consider using your own tile server or Mapbox
    servers = ['a', 'b', 'c']
    server = random.choice(servers)
    return f"https://{server}.tile.openstreetmap.org/{zoom}/{xtile}/{ytile}.png"


def download_tile(xtile, ytile, zoom, cache_dir, retry=3):
    """Download a single tile and cache it."""
    cache_path = Path(cache_dir) / f"{zoom}_{xtile}_{ytile}.png"
    
    if cache_path.exists():
        return cache_path
    
    url = get_tile_url(xtile, ytile, zoom)
    
    for attempt in range(retry):
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'MapCompressionModel/1.0'
            })
            response.raise_for_status()
            
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                f.write(response.content)
            return cache_path
        except Exception as e:
            if attempt == retry - 1:
                print(f"Failed to download tile {zoom}/{xtile}/{ytile}: {e}")
                return None
            continue
    
    return None


def get_tiles_in_region(min_lat, max_lat, min_lon, max_lon, zoom):
    """Get all tile coordinates in a bounding box."""
    n = 2 ** zoom
    max_tile_coord = n - 1
    
    # Clamp coordinates to valid range
    min_tile = deg2num(max_lat, min_lon, zoom)
    max_tile = deg2num(min_lat, max_lon, zoom)
    
    # Clamp to valid tile coordinate ranges [0, 2^zoom - 1]
    min_x = max(0, min(min_tile[0], max_tile[0]))
    max_x = min(max_tile_coord, max(min_tile[0], max_tile[0]))
    min_y = max(0, min(min_tile[1], max_tile[1]))
    max_y = min(max_tile_coord, max(min_tile[1], max_tile[1]))
    
    # Handle full globe case
    is_full_globe = (min_lat <= -90 and max_lat >= 90 and 
                     min_lon <= -180 and max_lon >= 180)
    if is_full_globe:
        min_x, max_x = 0, max_tile_coord
        min_y, max_y = 0, max_tile_coord
    
    tiles = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            tiles.append((x, y))
    return tiles


def load_tile_image(tile_path, tile_size):
    """Load and resize a tile image."""
    if tile_path is None or not os.path.exists(tile_path):
        return None
    
    img = Image.open(tile_path).convert('RGB')
    img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def lat_lon_to_normalized(lat, lon, zoom, min_lat, max_lat, min_lon, max_lon):
    """Normalize lat/lon/zoom to [0, 1] range for model input."""
    norm_lat = (lat - min_lat) / (max_lat - min_lat) if max_lat != min_lat else 0.5
    norm_lon = (lon - min_lon) / (max_lon - min_lon) if max_lon != min_lon else 0.5
    norm_zoom = zoom / 18.0  # Assuming max zoom of 18
    return np.array([norm_lat, norm_lon, norm_zoom], dtype=np.float32)


def get_tile_center_lat_lon(xtile, ytile, zoom):
    """Get the center lat/lon of a tile."""
    lat, lon = num2deg(xtile, ytile, zoom)
    # Get center by adding half tile
    n = 2.0 ** zoom
    center_lat, center_lon = num2deg(xtile + 0.5, ytile + 0.5, zoom)
    return center_lat, center_lon
