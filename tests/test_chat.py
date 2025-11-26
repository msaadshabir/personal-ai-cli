#!/usr/bin/env python3
"""
Unit tests for input validation in chat.py
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestInputValidation:
    """Test cases for input validation."""
    
    def test_empty_input(self):
        """Test that empty input is rejected."""
        from chat import validate_input
        
        assert validate_input("") == False
        assert validate_input("   ") == False
        assert validate_input("\n\t") == False
    
    def test_valid_input(self):
        """Test that valid input is accepted."""
        from chat import validate_input
        
        assert validate_input("Hello, how are you?") == True
        assert validate_input("What is my name?") == True
        assert validate_input("Tell me about my skills") == True
    
    def test_long_input(self):
        """Test that overly long input is rejected."""
        from chat import validate_input
        
        long_text = "a" * 1001
        assert validate_input(long_text) == False
        
        # Just under limit should pass
        valid_text = "a" * 999
        assert validate_input(valid_text) == True
    
    def test_custom_max_length(self):
        """Test custom max length parameter."""
        from chat import validate_input
        
        text = "a" * 50
        assert validate_input(text, max_length=100) == True
        assert validate_input(text, max_length=40) == False
    
    def test_script_injection(self):
        """Test that script injection is blocked."""
        from chat import validate_input
        
        assert validate_input("<script>alert('xss')</script>") == False
        assert validate_input("<SCRIPT>alert('xss')</SCRIPT>") == False
    
    def test_javascript_urls(self):
        """Test that JavaScript URLs are blocked."""
        from chat import validate_input
        
        assert validate_input("javascript:alert('xss')") == False
        assert validate_input("JAVASCRIPT:void(0)") == False
    
    def test_data_urls(self):
        """Test that data URLs are blocked."""
        from chat import validate_input
        
        assert validate_input("data:text/html,<script>alert('xss')</script>") == False
    
    def test_vbscript(self):
        """Test that VBScript is blocked."""
        from chat import validate_input
        
        assert validate_input("vbscript:msgbox('xss')") == False
    
    def test_normal_text_with_keywords(self):
        """Test that normal text containing blocked keywords in different context passes."""
        from chat import validate_input
        
        # These should pass because they don't match the dangerous patterns
        assert validate_input("Tell me about data analysis") == True
        assert validate_input("I love JavaScript programming") == True


class TestDatabaseCheck:
    """Test cases for database validation."""
    
    def test_missing_database(self, tmp_path):
        """Test that missing database is detected."""
        from chat import check_db
        
        non_existent = str(tmp_path / "nonexistent_db")
        assert check_db(non_existent) == False
    
    def test_existing_database(self, tmp_path):
        """Test that existing database is detected."""
        from chat import check_db
        
        # Create a directory to simulate database
        db_path = tmp_path / "test_db"
        db_path.mkdir()
        
        assert check_db(str(db_path)) == True
