#!/usr/bin/env python3
import json
from pathlib import Path





def load_existing_chunks():
    """Load existing chunks from lucassen directory structure."""
    lucassen_dir = Path("lucassen")
    if not lucassen_dir.exists():
        return {}
    
    # Find the first model directory with chunks.json
    for model_dir in lucassen_dir.iterdir():
        if model_dir.is_dir():
            chunks_file = model_dir / "chunks.json"
            if chunks_file.exists():
                try:
                    with open(chunks_file, 'r') as f:
                        chunks = json.load(f)
                    print(f"Loaded {len(chunks)} chunks from {model_dir.name}")
                    return chunks
                except Exception as e:
                    print(f"Error loading chunks from {model_dir.name}: {e}")
                    continue
    
    return {}

def main():
    # Load existing chunks instead of processing from CSV
    chunks = load_existing_chunks()
    
    if not chunks:
        print("No existing chunks found. Please run chunking first.")
        return
    
    # Save combined chunks for analyze_lucassen.py compatibility
    with open("lucassen_chunks.json", 'w') as f:
        json.dump(chunks, f, indent=2)
    
    print(f"Created lucassen_chunks.json with {len(chunks)} chunks")

if __name__ == "__main__":
    main()