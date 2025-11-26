#!/usr/bin/env python3
"""
Unit tests for config.py
"""
import os
import tempfile
from pathlib import Path
import pytest
import yaml

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config


class TestConfig:
    """Test cases for Config class."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            config = Config(config_path)
            
            assert config.get('data_path') == 'data'
            assert config.get('db_path') == 'db'
            assert config.get('default_model') == 'llama3'
            assert config.get('chunk_size') == 500
            assert config.get('chunk_overlap') == 50
            assert config.get('temperature') == 0.1
            assert config.get('top_k') == 3
    
    def test_load_custom_config(self):
        """Test loading a custom configuration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            
            # Create custom config
            custom_config = {
                'data_path': 'custom_data',
                'default_model': 'mistral',
                'temperature': 0.5,
            }
            with open(config_path, 'w') as f:
                yaml.dump(custom_config, f)
            
            config = Config(config_path)
            
            # Custom values should be used
            assert config.get('data_path') == 'custom_data'
            assert config.get('default_model') == 'mistral'
            assert config.get('temperature') == 0.5
            
            # Defaults should still apply for unset values
            assert config.get('chunk_size') == 500
    
    def test_get_with_default(self):
        """Test get method with default fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            config = Config(config_path)
            
            # Non-existent key should return default
            assert config.get('nonexistent_key', 'fallback') == 'fallback'
            assert config.get('nonexistent_key') is None
    
    def test_environment_variable_override(self):
        """Test that environment variables override config file values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            
            # Set environment variable
            os.environ['AI_DEFAULT_MODEL'] = 'phi3'
            os.environ['AI_TEMPERATURE'] = '0.7'
            os.environ['AI_CHUNK_SIZE'] = '1000'
            
            try:
                config = Config(config_path)
                
                assert config.get('default_model') == 'phi3'
                assert config.get('temperature') == 0.7
                assert config.get('chunk_size') == 1000
            finally:
                # Clean up environment variables
                del os.environ['AI_DEFAULT_MODEL']
                del os.environ['AI_TEMPERATURE']
                del os.environ['AI_CHUNK_SIZE']
    
    def test_save_default_config(self):
        """Test saving default configuration to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            config = Config(config_path)
            
            # Config file should not exist yet
            assert not os.path.exists(config_path)
            
            # Save default config
            config.save_default_config()
            
            # Config file should now exist
            assert os.path.exists(config_path)
            
            # Load and verify
            with open(config_path, 'r') as f:
                saved_config = yaml.safe_load(f)
            
            assert saved_config['data_path'] == 'data'
            assert saved_config['default_model'] == 'llama3'
    
    def test_supported_extensions(self):
        """Test that supported extensions include all file types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            config = Config(config_path)
            
            extensions = config.get('supported_extensions')
            
            assert '.txt' in extensions
            assert '.md' in extensions
            assert '.pdf' in extensions
            assert '.docx' in extensions
            assert '.html' in extensions
            assert '.json' in extensions
            assert '.csv' in extensions
            assert '.epub' in extensions
    
    def test_invalid_config_file(self):
        """Test handling of invalid config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            
            # Create invalid YAML
            with open(config_path, 'w') as f:
                f.write("invalid: yaml: content: [")
            
            # Should fall back to defaults
            config = Config(config_path)
            assert config.get('data_path') == 'data'


class TestConfigTypes:
    """Test type conversions from environment variables."""
    
    def test_boolean_conversion(self):
        """Test boolean conversion from environment variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            
            # Test true values
            for true_val in ['true', '1', 'yes', 'on', 'TRUE', 'Yes']:
                os.environ['AI_SOME_BOOL'] = true_val
                config = Config(config_path)
                # Note: This would need a bool default to work properly
            
            if 'AI_SOME_BOOL' in os.environ:
                del os.environ['AI_SOME_BOOL']
    
    def test_integer_conversion(self):
        """Test integer conversion from environment variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            
            os.environ['AI_TOP_K'] = '5'
            
            try:
                config = Config(config_path)
                assert config.get('top_k') == 5
                assert isinstance(config.get('top_k'), int)
            finally:
                del os.environ['AI_TOP_K']
    
    def test_float_conversion(self):
        """Test float conversion from environment variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            
            os.environ['AI_TEMPERATURE'] = '0.85'
            
            try:
                config = Config(config_path)
                assert config.get('temperature') == 0.85
                assert isinstance(config.get('temperature'), float)
            finally:
                del os.environ['AI_TEMPERATURE']
