#!/usr/bin/env python3
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent))

from llm_qus_analyzer.client import LLMClient
from llm_qus_analyzer.settings import Settings
from llm_qus_analyzer.chunker.models import QUSChunkerModel

def analyze_with_retry(chunker, client, model_idx, story, max_retries=5):
    """
    Analyze a story with exponential backoff retry logic for HTTP errors.
    
    Args:
        chunker: The QUSChunkerModel instance
        client: The LLMClient instance
        model_idx: Index of the model to use
        story: The user story to analyze
        max_retries: Maximum number of retry attempts
    
    Returns:
        tuple: (component, usage) if successful, (None, error_message) if failed
    """
    base_delay = 1  # Start with 1 second delay
    
    for attempt in range(max_retries):
        try:
            component, usage = chunker.analyze_single(client, model_idx, story)
            return component, usage
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a rate limit or HTTP error that we should retry
            if any(http_error in error_msg.lower() for http_error in ['http', '429', '500', '502', '503', '504', 'timeout', 'connection']):
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f" ⚠ HTTP error: {error_msg[:50]}... Retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                    continue
            
            # For all other errors or if we've exhausted retries, return the error
            return None, error_msg
    
    # If we've exhausted all retries
    return None, f"Failed after {max_retries} attempts"

def main():
    settings = Settings()
    settings.configure_paths_and_load(".env", "models.yaml")
    client = LLMClient(settings)
    chunker = QUSChunkerModel()
    
    stories = []
    with open("lucassen.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row and row[0].strip():
                stories.append(row[0].strip())
    
    chunks = {}
    errors = {}
    
    for i, story in enumerate(stories):
        print(f"{i+1}/{len(stories)}", end="")
        component, result = analyze_with_retry(chunker, client, 1, story)
        
        if component is not None:
            chunks[f"story_{i+1}"] = {
                "text": component.text,
                "role": component.role,
                "means": component.means,
                "ends": component.ends,
                "original_story": story,
                "template_data": {
                    "text": component.template.text,
                    "chunk": component.template.chunk,
                    "tail": component.template.tail,
                    "order": component.template.order
                }
            }
            print(f" ✓ {component.role}")
        else:
            errors[f"story_{i+1}"] = {
                "error": result,
                "original_story": story
            }
            print(f" ✗ {result[:50]}...")
    
    with open("lucassen_chunks.json", 'w') as f:
        json.dump(chunks, f, indent=2)
    
    if errors:
        with open("lucassen_errors.json", 'w') as f:
            json.dump(errors, f, indent=2)
    
    print(f"Done: {len(chunks)}/{len(stories)} successful")
    if errors:
        print(f"Errors: {len(errors)}/{len(stories)}")

if __name__ == "__main__":
    main()