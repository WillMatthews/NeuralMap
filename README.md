# Map Compression Model

An experimental project to compress world map data into a small neural network model. Instead of loading and rendering traditional map tiles, this model can generate map tiles on-the-fly by querying coordinates (latitude, longitude, zoom level).

## Overview

This project uses a coordinate-based transformer model that learns to map geographic coordinates directly to RGB map tiles. The model is trained on OpenStreetMap tiles and can generate map tiles in real-time.

## Features

- **Coordinate-based MLP with Transformers**: Uses advanced transformer architecture to learn the mapping from lat/lon/zoom to map tiles
- **Real-time tile generation**: Generate map tiles on-demand without storing large tile databases
- **Configurable training**: Support for different zoom levels (0-10+) and tile sizes (256x256, 512x512)
- **Web visualization**: Compare generated tiles side-by-side with OpenStreetMap reference tiles
- **TensorBoard integration**: Monitor training progress with detailed metrics

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- PyTorch 2.0+
- CUDA-capable GPU (recommended, tested on RTX 2060 Super)
- ~8GB VRAM for training

## Setup

1. Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies and set up the project:
```bash
task setup
```

Or manually:
```bash
uv sync
mkdir -p data/tiles data/cache models logs
```

All Python commands should be run through `uv`:
```bash
uv run python scripts/train.py
```

Or use the task runner (which uses `uv` under the hood):
```bash
task train
```

2. Configure the project by editing `config.yaml`:
   - Set your training region (lat/lon bounds)
   - Adjust model architecture parameters
   - Configure training hyperparameters

## Usage

### 1. Download Training Data

Download OpenStreetMap tiles for your region:
```bash
task download
```

This will download tiles based on the region and zoom levels specified in `config.yaml`.

### 2. Train the Model

Start training:
```bash
task train
```

Training progress will be logged to TensorBoard. View it with:
```bash
task tensorboard
```

Then open http://localhost:6006 in your browser.

### 3. Serve and Visualize

Start the web server to visualize generated tiles:
```bash
task serve
```

Open http://localhost:5000 in your browser to see a side-by-side comparison of:
- Left: OpenStreetMap reference tiles
- Right: Model-generated tiles

You can pan and zoom around the map to see how well the model generalizes.

## Project Structure

```
mappy/
├── src/
│   ├── model.py          # Coordinate-based transformer model
│   ├── dataset.py        # PyTorch dataset for map tiles
│   └── data_utils.py     # Utilities for downloading/processing tiles
├── scripts/
│   ├── download_tiles.py # Download OSM tiles
│   ├── train.py          # Training script
│   └── serve.py          # Web server for visualization
├── config.yaml           # Configuration file
├── Taskfile.yml          # Task runner configuration
└── requirements.txt      # Python dependencies
```

## Configuration

Edit `config.yaml` to customize:

- **Model architecture**: Hidden dimensions, number of layers, attention heads
- **Training**: Batch size, learning rate, number of epochs
- **Data**: Zoom levels, tile size, number of tiles per zoom
- **Region**: Bounding box for training data (start small for faster iteration)

## Model Architecture

The model uses:
- **Input**: Normalized (lat, lon, zoom) coordinates
- **Positional Encoding**: Sinusoidal encoding for coordinate features
- **Transformer Encoder**: Multi-head self-attention layers
- **Spatial Decoder**: CNN-based upsampling to generate RGB tiles

## Performance Targets

- Model size: 100s of MB
- Inference: Real-time tile generation
- Quality: Visually similar to reference OSM tiles

## Notes

- This is an experimental project - results may vary
- Start with a small region for faster iteration
- The model learns the style and features of the training region
- Higher zoom levels require more training data and model capacity

## License

This project uses OpenStreetMap tiles, which are licensed under ODbL. Please respect OSM's usage policy when downloading tiles.
