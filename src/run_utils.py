"""Utilities for managing training runs with human-friendly IDs and metadata."""
import json
import random
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


# Human-friendly adjectives and nouns for run IDs
ADJECTIVES = [
    "happy", "swift", "bright", "calm", "bold", "clever", "daring", "eager",
    "gentle", "jolly", "keen", "lively", "merry", "noble", "proud", "quick",
    "radiant", "serene", "tidy", "vivid", "witty", "zesty", "brave", "calm",
    "dapper", "epic", "fancy", "grand", "heroic", "jolly", "kind", "lucky"
]

NOUNS = [
    "dolphin", "eagle", "falcon", "tiger", "lion", "wolf", "bear", "hawk",
    "shark", "whale", "panda", "koala", "otter", "seal", "fox", "deer",
    "rabbit", "squirrel", "badger", "beaver", "lynx", "bison", "moose",
    "elk", "crane", "swan", "raven", "owl", "jaguar", "leopard", "cheetah"
]


def generate_run_id() -> str:
    """
    Generate a human-friendly random run ID.
    Format: adjective-noun-number (e.g., "happy-dolphin-42")
    """
    adjective = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    number = random.randint(1, 999)
    return f"{adjective}-{noun}-{number}"


def get_git_commit_hash(repo_path: Optional[Path] = None) -> Optional[str]:
    """
    Get the current git commit hash.
    
    Args:
        repo_path: Path to git repository root. If None, uses current directory.
    
    Returns:
        Commit hash string, or None if git is not available or not in a git repo.
    """
    if repo_path is None:
        repo_path = Path.cwd()
    
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_status(repo_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get git repository status information.
    
    Returns:
        Dictionary with commit hash, branch, and dirty status.
    """
    if repo_path is None:
        repo_path = Path.cwd()
    
    status = {
        'commit_hash': None,
        'branch': None,
        'is_dirty': False
    }
    
    try:
        # Get commit hash
        commit_result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        status['commit_hash'] = commit_result.stdout.strip()
        
        # Get branch name
        branch_result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        status['branch'] = branch_result.stdout.strip()
        
        # Check if working directory is dirty
        diff_result = subprocess.run(
            ['git', 'diff', '--quiet'],
            cwd=repo_path,
            capture_output=True
        )
        status['is_dirty'] = diff_result.returncode != 0
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return status


def save_run_metadata(
    run_id: str,
    config: Dict[str, Any],
    runs_dir: Path,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save run metadata to a JSON file.
    
    Args:
        run_id: Human-friendly run ID
        config: Training configuration dictionary
        runs_dir: Directory to save metadata files
        additional_metadata: Any additional metadata to include
    
    Returns:
        Path to the saved metadata file
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Get git information
    git_info = get_git_status()
    
    # Prepare metadata
    metadata = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'git': git_info,
        'config': config,
    }
    
    if additional_metadata:
        metadata.update(additional_metadata)
    
    # Save to JSON file
    metadata_file = runs_dir / f"{run_id}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_file


def load_run_metadata(run_id: str, runs_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load run metadata from a JSON file.
    
    Args:
        run_id: Run ID to load
        runs_dir: Directory containing metadata files
    
    Returns:
        Metadata dictionary, or None if not found
    """
    metadata_file = runs_dir / f"{run_id}.json"
    if not metadata_file.exists():
        return None
    
    with open(metadata_file, 'r') as f:
        return json.load(f)
