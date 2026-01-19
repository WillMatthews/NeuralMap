# Training Run Metadata

This directory contains metadata for each training run in JSON format.

Each run gets a human-friendly ID (e.g., `happy-dolphin-42`) and a corresponding JSON file (`happy-dolphin-42.json`) that contains:

- **run_id**: The human-friendly identifier for this run
- **timestamp**: When the run started (ISO format)
- **git**: Git repository information
  - `commit_hash`: The git commit hash used for this run
  - `branch`: The git branch name
  - `is_dirty`: Whether the working directory had uncommitted changes
- **config**: The full training configuration used
- **device**: The device used for training (CPU/GPU)
- **logs_dir**: Path to TensorBoard logs for this run
- **models_dir**: Path where model checkpoints are saved

## Example

```json
{
  "run_id": "happy-dolphin-42",
  "timestamp": "2024-01-15T10:30:00.123456",
  "git": {
    "commit_hash": "abc123def456...",
    "branch": "main",
    "is_dirty": false
  },
  "config": {
    "model": { ... },
    "training": { ... },
    ...
  },
  "device": "cuda",
  "logs_dir": "logs/happy-dolphin-42",
  "models_dir": "models"
}
```

## Usage

To view TensorBoard for a specific run:
```bash
tensorboard --logdir logs/happy-dolphin-42
```

To load metadata for a run:
```python
from src.run_utils import load_run_metadata
from pathlib import Path

metadata = load_run_metadata("happy-dolphin-42", Path("runs"))
print(metadata['git']['commit_hash'])
```
