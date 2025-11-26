#!/usr/bin/env python3
"""
Configuration management for personal-ai-cli.
"""
import os
from pathlib import Path
from typing import Dict, Any
import yaml

class Config:
    """Configuration manager for the personal AI CLI."""

    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = Path(config_file)
        self._defaults = {
            "data_path": "data",
            "db_path": "db",
            "default_model": "llama3",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "embedding_model": "all-MiniLM-L6-v2",
            "max_tokens": 512,
            "temperature": 0.1,
            "top_k": 3,
            "supported_extensions": [".txt", ".md", ".pdf", ".docx", ".html", ".json", ".csv", ".epub"]
        }
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file, falling back to defaults."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = yaml.safe_load(f) or {}
                # Merge user config with defaults
                config = self._defaults.copy()
                config.update(user_config)
                return config
            except Exception as e:
                print(f"[WARNING] Warning: Could not load config file {self.config_file}: {e}")
                print("Using default configuration.")
                return self._defaults.copy()
        else:
            return self._defaults.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        # First check environment variables (override config file)
        env_key = f"AI_{key.upper()}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            # Try to convert to appropriate type using defaults as reference
            if isinstance(self._defaults.get(key), bool):
                return env_value.lower() in ('true', '1', 'yes', 'on')
            elif isinstance(self._defaults.get(key), int):
                try:
                    return int(env_value)
                except ValueError:
                    pass
            elif isinstance(self._defaults.get(key), float):
                try:
                    return float(env_value)
                except ValueError:
                    pass
            return env_value

        return self._config.get(key, default)

    def save_default_config(self):
        """Save the default configuration to file if it doesn't exist."""
        if not self.config_file.exists():
            try:
                with open(self.config_file, 'w') as f:
                    yaml.dump(self._defaults, f, default_flow_style=False, sort_keys=False)
                print(f"[SUCCESS] Created default configuration file: {self.config_file}")
            except Exception as e:
                print(f"[WARNING] Warning: Could not create config file {self.config_file}: {e}")

# Global config instance
config = Config()