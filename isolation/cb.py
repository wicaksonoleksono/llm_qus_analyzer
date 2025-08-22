#!/usr/bin/env python3
"""
Per-Criteria Chunker - Chunks stories for specific PT criteria
Loads criteria-specific test data and saves chunks to pre-chunked/ folder.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict

sys.path.append(str(Path(__file__).parent.parent))

from llm_qus_analyzer.client import LLMClient
from llm_qus_analyzer.settings import Settings
from llm_qus_analyzer.chunker import QUSChunkerModel

# Available criteria (matches the split test data files)
AVAILABLE_CRITERIA = [
    "complete", "conceptually-sound", "conflict-free", "estimatable", 
    "full-sentence", "independent", "problem-oriented", "unambiguous", "unique", "unknown"
]

SET_CATEGORIES = ["conflict-free", "complete", "independent", "unique"]
INDIVIDUAL_CATEGORIES = ["conceptually-sound", "problem-oriented", "full-sentence", "estimatable", "unambiguous"]


def load_test_data(criteria: str) -> List[Dict]:
    """Load test data for specific criteria."""
    test_data_path = Path(__file__).parent / "testdata" / "by-criteria" / f"{criteria}.json"
    
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")
    
    with open(test_data_path, 'r') as f:
        return json.load(f)


def load_existing_chunks(file_path: Path) -> Dict:
    """Load existing chunks from JSON file if it exists."""
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}


def save_chunks(chunks: Dict, file_path: Path):
    """Save chunks to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(chunks, f, indent=2, default=str)


def chunk_criteria_stories(client: LLMClient, chunker: QUSChunkerModel, 
                          test_data: List[Dict], criteria: str, existing_chunks: Dict = None) -> Dict:
    """Chunk all stories for a specific criteria, skipping already chunked ones."""
    
    chunks = existing_chunks.copy() if existing_chunks else {}
    stories_to_chunk = []
    story_metadata = {}
    
    print(f"Analyzing {len(test_data)} items for '{criteria}' criteria")
    
    # Collect all unique stories and their metadata
    for item in test_data:
        if "stories" in item and len(item["stories"]) == 2:
            # Pairwise stories (set analyzers)
            stories = item["stories"]
            violation = item.get("violation", "")
            
            for idx, story in enumerate(stories):
                story_hash = str(hash(story))
                
                # Store metadata for this story
                if story_hash not in story_metadata:
                    story_metadata[story_hash] = {
                        "pt_categories": set(),
                        "expected_violations": set(),
                        "analysis_types": set(),
                        "pair_contexts": [],
                        "individual_contexts": []
                    }
                
                story_metadata[story_hash]["pt_categories"].add(criteria)
                story_metadata[story_hash]["expected_violations"].add(violation)
                story_metadata[story_hash]["analysis_types"].add("pairwise")
                
                # Store pair context
                other_story_hash = str(hash(stories[1 - idx]))
                story_metadata[story_hash]["pair_contexts"].append({
                    "pt": criteria,
                    "violation": violation,
                    "position_in_pair": idx + 1,
                    "paired_with_hash": other_story_hash
                })
                
                if story not in [s[0] for s in stories_to_chunk] and story_hash not in chunks:
                    stories_to_chunk.append((story, story_hash))
        
        elif "story" in item:
            # Individual stories (individual analyzers)
            story = item["story"]
            story_hash = str(hash(story))
            violation = item.get("violation", "")
            
            # Store metadata for this story
            if story_hash not in story_metadata:
                story_metadata[story_hash] = {
                    "pt_categories": set(),
                    "expected_violations": set(),
                    "analysis_types": set(),
                    "pair_contexts": [],
                    "individual_contexts": []
                }
            
            story_metadata[story_hash]["pt_categories"].add(criteria)
            story_metadata[story_hash]["expected_violations"].add(violation)
            story_metadata[story_hash]["analysis_types"].add("individual")
            story_metadata[story_hash]["individual_contexts"].append({
                "pt": criteria,
                "violation": violation
            })
            
            if story not in [s[0] for s in stories_to_chunk] and story_hash not in chunks:
                stories_to_chunk.append((story, story_hash))
    
    print(f"Found {len(stories_to_chunk)} unique stories to chunk")
    
    # Chunk each story
    for i, (story, story_hash) in enumerate(stories_to_chunk):
        print(f"Chunking story {i+1}/{len(stories_to_chunk)}: {story[:50]}...")
        
        try:
            component, usage = chunker.analyze_single(client, 0, story, f"story_{story_hash}")
            
            # Get metadata for this story
            metadata = story_metadata.get(story_hash, {})
            
            # Convert sets to lists for JSON serialization
            serializable_metadata = {
                "pt_categories": list(metadata.get("pt_categories", set())),
                "expected_violations": list(metadata.get("expected_violations", set())),
                "analysis_types": list(metadata.get("analysis_types", set())),
                "pair_contexts": metadata.get("pair_contexts", []),
                "individual_contexts": metadata.get("individual_contexts", [])
            }
            
            # Store complete template data for proper reconstruction
            template_data = {
                "text": component.template.text,
                "chunk": component.template.chunk,
                "tail": component.template.tail,
                "order": component.template.order
            }
            
            # Convert component to serializable dict with enhanced tags
            chunk_data = {
                "text": component.text,
                "role": component.role,
                "means": component.means,
                "ends": component.ends,
                "id": component.id,
                "original_story": story,
                "story_hash": story_hash,
                
                # Store complete template for proper reconstruction
                "template_data": template_data,
                
                # Enhanced tagging for targeted analysis
                "tags": serializable_metadata,
                
                "usage": {
                    "duration": usage.duration,
                    "num_token_in": usage.num_token_in,
                    "num_token_out": usage.num_token_out
                },
                
                # Quick access fields
                "primary_pt": serializable_metadata["pt_categories"][0] if serializable_metadata["pt_categories"] else "unknown",
                "primary_violation": serializable_metadata["expected_violations"][0] if serializable_metadata["expected_violations"] else "",
                "is_pairwise": "pairwise" in serializable_metadata["analysis_types"],
                "is_individual": "individual" in serializable_metadata["analysis_types"]
            }
            
            chunks[story_hash] = chunk_data
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error chunking story {i+1}: {e}")
            continue
    
    return chunks


def main():
    parser = argparse.ArgumentParser(description='Chunk stories by specific criteria')
    parser.add_argument('--criteria', '-c', type=str, 
                       help='Specific criteria to chunk (e.g., conceptually-sound, complete)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available criteria')
    parser.add_argument('--all', action='store_true',
                       help='Chunk all criteria (one by one)')
    
    args = parser.parse_args()
    
    # Configuration
    models_path = Path(__file__).parent.parent / "models.yaml"
    env_path = Path(__file__).parent.parent / ".env"
    pre_chunked_dir = Path(__file__).parent / "pre-chunked"
    
    # Handle --list flag
    if args.list:
        print("Available criteria:")
        for criteria in sorted(AVAILABLE_CRITERIA):
            criteria_type = "SET" if criteria in SET_CATEGORIES else "INDIVIDUAL"
            print(f"  {criteria} ({criteria_type})")
        return
    
    # Determine criteria to process
    if args.all:
        criteria_to_process = AVAILABLE_CRITERIA
        print(f"🔄 Chunking all {len(criteria_to_process)} criteria")
    elif args.criteria:
        if args.criteria not in AVAILABLE_CRITERIA:
            print(f"❌ Error: '{args.criteria}' not found in available criteria")
            print(f"Available: {sorted(AVAILABLE_CRITERIA)}")
            return
        criteria_to_process = [args.criteria]
        print(f"🎯 Chunking single criteria: {args.criteria}")
    else:
        print("❌ Error: Must specify --criteria <name>, --all, or --list")
        return
    
    try:
        # Setup
        settings = Settings()
        settings.configure_paths_and_load(env_path if env_path.exists() else None, models_path)
        
        client = LLMClient(settings)
        chunker = QUSChunkerModel()
        
        print(f"Per-Criteria Chunker")
        print(f"Model: {client.names[0]}")
        print(f"Output directory: {pre_chunked_dir}")
        
        # Process each criteria
        for criteria in criteria_to_process:
            print(f"\n{'='*50}")
            print(f"CHUNKING CRITERIA: {criteria.upper()}")
            print(f"{'='*50}")
            
            # Load test data for this criteria
            test_data = load_test_data(criteria)
            
            # Load existing chunks (if any)
            chunks_file = pre_chunked_dir / f"{criteria}_chunks.json"
            existing_chunks = load_existing_chunks(chunks_file)
            print(f"Found {len(existing_chunks)} existing chunks")
            
            # Chunk stories for this criteria (skipping already chunked ones)
            chunks = chunk_criteria_stories(client, chunker, test_data, criteria, existing_chunks)
            
            # Save chunks
            save_chunks(chunks, chunks_file)
            
            new_chunks_count = len(chunks) - len(existing_chunks)
            print(f"\n✅ Processed {len(chunks)} total stories for '{criteria}' ({new_chunks_count} new, {len(existing_chunks)} existing)")
            print(f"💾 Saved: {chunks_file}")
        
        print(f"\n{'='*60}")
        print(f"CHUNKING COMPLETE!")
        print(f"{'='*60}")
        print(f"✅ Processed {len(criteria_to_process)} criteria")
        print(f"✅ Chunks saved in: {pre_chunked_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()