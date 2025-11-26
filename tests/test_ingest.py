#!/usr/bin/env python3
"""
Unit tests for ingest.py
"""
import os
import sys
import tempfile
import hashlib
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFileHashing:
    """Test cases for file hashing functionality."""
    
    def test_compute_file_hash(self, tmp_path):
        """Test that file hashing works correctly."""
        from ingest import compute_file_hash
        
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")
        
        hash1 = compute_file_hash(test_file)
        
        # Hash should be consistent
        hash2 = compute_file_hash(test_file)
        assert hash1 == hash2
        
        # Hash should be a valid SHA256 (64 hex characters)
        assert len(hash1) == 64
        assert all(c in '0123456789abcdef' for c in hash1)
    
    def test_different_files_different_hashes(self, tmp_path):
        """Test that different files produce different hashes."""
        from ingest import compute_file_hash
        
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        
        file1.write_text("Content 1")
        file2.write_text("Content 2")
        
        hash1 = compute_file_hash(file1)
        hash2 = compute_file_hash(file2)
        
        assert hash1 != hash2
    
    def test_modified_file_different_hash(self, tmp_path):
        """Test that modifying a file changes its hash."""
        from ingest import compute_file_hash
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content")
        
        hash1 = compute_file_hash(test_file)
        
        # Modify the file
        test_file.write_text("Modified content")
        
        hash2 = compute_file_hash(test_file)
        
        assert hash1 != hash2


class TestHashStorage:
    """Test cases for hash storage functionality."""
    
    def test_save_and_load_hashes(self, tmp_path, monkeypatch):
        """Test saving and loading file hashes."""
        from ingest import save_file_hashes, load_file_hashes, HASH_FILE
        
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        test_hashes = {
            "/path/to/file1.txt": "abc123",
            "/path/to/file2.md": "def456",
        }
        
        save_file_hashes(test_hashes)
        
        # Hash file should exist
        assert os.path.exists(HASH_FILE)
        
        # Load and verify
        loaded_hashes = load_file_hashes()
        assert loaded_hashes == test_hashes
    
    def test_load_missing_hash_file(self, tmp_path, monkeypatch):
        """Test loading when hash file doesn't exist."""
        from ingest import load_file_hashes
        
        monkeypatch.chdir(tmp_path)
        
        hashes = load_file_hashes()
        assert hashes == {}


class TestDataDirectoryValidation:
    """Test cases for data directory validation."""
    
    def test_missing_directory(self, tmp_path):
        """Test validation of missing directory."""
        from ingest import validate_data_directory
        
        non_existent = str(tmp_path / "nonexistent")
        assert validate_data_directory(non_existent) == False
    
    def test_empty_directory(self, tmp_path):
        """Test validation of empty directory."""
        from ingest import validate_data_directory
        
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        assert validate_data_directory(str(empty_dir)) == False
    
    def test_directory_with_supported_files(self, tmp_path):
        """Test validation of directory with supported files."""
        from ingest import validate_data_directory
        
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create a supported file
        (data_dir / "test.md").write_text("# Test")
        
        assert validate_data_directory(str(data_dir)) == True
    
    def test_directory_with_unsupported_files_only(self, tmp_path):
        """Test validation of directory with only unsupported files."""
        from ingest import validate_data_directory
        
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create an unsupported file
        (data_dir / "test.xyz").write_text("unsupported")
        
        assert validate_data_directory(str(data_dir)) == False
    
    def test_file_instead_of_directory(self, tmp_path):
        """Test validation when path is a file, not directory."""
        from ingest import validate_data_directory
        
        file_path = tmp_path / "file.txt"
        file_path.write_text("I'm a file")
        
        assert validate_data_directory(str(file_path)) == False


class TestDocumentLoading:
    """Test cases for document loading functionality."""
    
    def test_load_markdown_file(self, tmp_path):
        """Test loading a markdown file."""
        from ingest import load_single_file
        
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello\n\nThis is a test.")
        
        docs = load_single_file(md_file)
        
        assert len(docs) == 1
        assert "Hello" in docs[0].page_content
    
    def test_load_text_file(self, tmp_path):
        """Test loading a text file."""
        from ingest import load_single_file
        
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Plain text content")
        
        docs = load_single_file(txt_file)
        
        assert len(docs) == 1
        assert "Plain text content" in docs[0].page_content
    
    def test_load_unsupported_file(self, tmp_path):
        """Test loading an unsupported file type."""
        from ingest import load_single_file
        
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("Unsupported content")
        
        docs = load_single_file(unsupported_file)
        
        # Should return empty list for unsupported files
        assert docs == []


class TestChunking:
    """Test cases for text chunking functionality."""
    
    def test_create_chunks(self, tmp_path):
        """Test creating chunks from documents."""
        from ingest import create_chunks
        from langchain_core.documents import Document
        
        # Create a document with enough content to be chunked
        long_content = "This is a test. " * 100
        documents = [Document(page_content=long_content)]
        
        chunks = create_chunks(documents)
        
        # Should create multiple chunks
        assert len(chunks) > 1
    
    def test_empty_documents(self):
        """Test chunking with no documents."""
        from ingest import create_chunks
        
        chunks = create_chunks([])
        
        assert chunks == []
    
    def test_short_document_single_chunk(self, tmp_path):
        """Test that short documents result in single chunk."""
        from ingest import create_chunks
        from langchain_core.documents import Document
        
        short_content = "Short text."
        documents = [Document(page_content=short_content)]
        
        chunks = create_chunks(documents)
        
        # Short content should remain in single chunk
        assert len(chunks) == 1
