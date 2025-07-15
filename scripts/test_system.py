#!/usr/bin/env python3

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config.settings import Settings
from scripts.demo import RAGPipeline


def test_basic_functionality():
    print("Testing basic RAG functionality...")
    
    settings = Settings()
    
    if not Path(settings.DATA_RAW_DIR).exists():
        print(f"Creating test data directory: {settings.DATA_RAW_DIR}")
        Path(settings.DATA_RAW_DIR).mkdir(parents=True, exist_ok=True)
    
    test_files = list(Path(settings.DATA_RAW_DIR).glob("*.md"))
    if not test_files:
        print("No markdown files found in data/raw/")
        print("Creating a test file...")
        test_content = """# Programming Concepts

## Variables
Variables are containers for storing data values. In programming, a variable is a symbolic name for a memory location.

## Functions
Functions are reusable blocks of code that perform specific tasks. They help organize code and avoid repetition.

## Object-Oriented Programming
Object-oriented programming (OOP) is a programming paradigm based on the concept of objects, which contain data and code.
"""
        test_file = Path(settings.DATA_RAW_DIR) / "test_programming.md"
        test_file.write_text(test_content)
        print(f"Created test file: {test_file}")
    
    try:
        pipeline = RAGPipeline(settings)
        print("✓ Pipeline initialized successfully")
        
        print("Building index...")
        result = pipeline.build_index()
        print(f"✓ Index built: {result}")
        
        print("Testing query...")
        query_result = pipeline.query("What are variables in programming?")
        print(f"✓ Query successful")
        print(f"Answer: {query_result['answer'][:100]}...")
        
        stats = pipeline.get_database_stats()
        print(f"✓ Database stats: {stats}")
        
        print("\n🎉 All tests passed! The RAG system is working correctly.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
