# src/utils/config.py
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path: str | Path) -> DictConfig:
    """Load a YAML config file and return as OmegaConf DictConfig."""
    return OmegaConf.load(config_path)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configs. Later configs override earlier ones."""
    return OmegaConf.merge(*configs)


def config_to_dict(cfg: DictConfig) -> dict:
    """Convert OmegaConf config to plain Python dict."""
    return OmegaConf.to_container(cfg, resolve=True)