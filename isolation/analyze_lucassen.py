#!/usr/bin/env python3
import json
import sys
import csv
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from llm_qus_analyzer.client import LLMClient
from llm_qus_analyzer.settings import Settings
from llm_qus_analyzer.chunker.models import QUSComponent
from llm_qus_analyzer.chunker.parser import Template
from llm_qus_analyzer.individual.atomic import AtomicAnalyzer
from llm_qus_analyzer.individual.minimal import MinimalAnalyzer
from llm_qus_analyzer.set.unique import UniqueAnalyzer
from llm_qus_analyzer.set.conflict_free import ConflictFreeAnalyzer

def create_component(chunk_data, story_id):
    template = Template(
        text=chunk_data["template_data"]["text"],
        chunk=chunk_data["template_data"]["chunk"],
        tail=chunk_data["template_data"]["tail"],
        order=chunk_data["template_data"]["order"]
    )
    return QUSComponent(
        text=chunk_data["text"],
        role=chunk_data["role"],
        means=chunk_data["means"],
        ends=chunk_data["ends"],
        template=template,
        id=story_id,
        original_text=chunk_data["original_story"]
    )

def analyze_with_retry(analyzer, client, model_idx, *args, max_retries=5):
    """
    Analyze with exponential backoff retry logic for HTTP errors.
    
    Args:
        analyzer: The analyzer instance (AtomicAnalyzer, MinimalAnalyzer, etc.)
        client: The LLMClient instance
        model_idx: Index of the model to use
        *args: Arguments to pass to the analyzer method
        max_retries: Maximum number of retry attempts
    
    Returns:
        tuple: (result, usage) if successful, (None, error_message) if failed
    """
    base_delay = 1  # Start with 1 second delay
    
    # Determine which method to call based on the analyzer type
    method = getattr(analyzer, 'run', None)
    if method is None:
        # For set analyzers, we use analyze_pairwise
        method = getattr(analyzer, 'analyze_pairwise', None)
    
    if method is None:
        return None, "Analyzer does not have a valid method to call"
    
    for attempt in range(max_retries):
        try:
            result = method(client, model_idx, *args)
            return result
        except json.JSONDecodeError as e:
            error_msg = str(e)
            print(f"\n  JSON Decode Error:")
            print(f"    Error: {error_msg}")
            print(f"    Position: line {e.lineno} column {e.colno} (char {e.pos})")
            # Try to get more context about what was being parsed
            try:
                # If the method stores the last response, we can inspect it
                if hasattr(analyzer, 'last_response') and analyzer.last_response:
                    print(f"    Last response snippet: {analyzer.last_response[:200]}...")
            except:
                pass
            return None, f"JSON parsing error: {error_msg}"
        except Exception as e:
            error_msg = str(e)
            
            # Print more detailed error information for debugging
            print(f"\n  Detailed error info:")
            print(f"    Error type: {type(e).__name__}")
            print(f"    Error message: {error_msg}")
            
            # Special handling for the specific error we're seeing
            if "Expecting value: line 1 column 2 (char 1)" in error_msg:
                print(f"    This suggests an empty or malformed JSON response from the LLM")
                try:
                    # If the method stores the last response, we can inspect it
                    if hasattr(analyzer, 'last_response') and analyzer.last_response:
                        print(f"    Last response: {repr(analyzer.last_response[:100])}")
                except:
                    pass
            
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

def load_existing_results():
    """Load existing results from lucassen_analysis.json if it exists."""
    try:
        with open("lucassen_analysis.json", 'r') as f:
            content = f.read()
            if not content.strip():
                print("Warning: lucassen_analysis.json is empty")
                return {}
            return json.loads(content)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Could not load existing results due to JSON error: {e}")
        print("Starting with fresh results...")
        return {}

def save_results(results):
    """Save results to lucassen_analysis.json."""
    try:
        with open("lucassen_analysis.json", 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving results: {e}")
        # Try to save a backup
        try:
            with open("lucassen_analysis_backup.json", 'w') as f:
                json.dump(results, f, indent=2)
            print("Saved backup to lucassen_analysis_backup.json")
        except Exception as e2:
            print(f"Error saving backup: {e2}")

def main():
    settings = Settings()
    settings.configure_paths_and_load(".env", "models.yaml")
    client = LLMClient(settings)
    
    with open("lucassen_chunks.json", 'r') as f:
        chunks = json.load(f)
    
    individual_analyzers = {"atomic": AtomicAnalyzer, "minimal": MinimalAnalyzer}
    set_analyzers = {"unique": UniqueAnalyzer, "conflict_free": ConflictFreeAnalyzer}
    
    # Load existing results to avoid re-analyzing
    results = load_existing_results()
    
    # Individual analysis
    for name, analyzer in individual_analyzers.items():
        # Skip if already analyzed
        if name in results and results[name]:
            print(f"Skipping {name} - already analyzed")
            continue
            
        print(f"Analyzing {name}")
        analyzer_results = []
        errors = []
        
        for i, (chunk_id, chunk_data) in enumerate(chunks.items()):
            print(f"  {i+1}/{len(chunks)}", end="")
            component = create_component(chunk_data, chunk_id)
            result = analyze_with_retry(analyzer(), client, 1, component)
            
            if result[0] is not None:
                violations, usage = result
                analyzer_results.append({
                    "story_id": chunk_id,
                    "story": chunk_data["original_story"],
                    "violations": [{"issue": v.issue, "suggestion": v.suggestion} for v in violations],
                    "violation_count": len(violations)
                })
                print(f" ✓ {len(violations)} violations")
            else:
                errors.append({
                    "story_id": chunk_id,
                    "story": chunk_data["original_story"],
                    "error": result[1]
                })
                print(f" ✗ Error: {result[1][:50]}...")
        
        results[name] = analyzer_results
        if errors:
            results[f"{name}_errors"] = errors
            
        # Save progress after each analyzer
        save_results(results)
    
    # Set pairwise analysis (commutative - only A:B, not B:A)
    components = [create_component(chunk_data, chunk_id) for chunk_id, chunk_data in chunks.items()]
    
    # Create sets of already analyzed pairs for each analyzer
    analyzed_pairs = {}
    for name in set_analyzers.keys():
        if name in results and results[name]:
            analyzed_pairs[name] = set(result["pair"] for result in results[name])
        else:
            analyzed_pairs[name] = set()
        if f"{name}_errors" in results and results[f"{name}_errors"]:
            analyzed_pairs[name].update(result["pair"] for result in results[f"{name}_errors"])
    
    for name, analyzer in set_analyzers.items():
        total_pairs = len(components) * (len(components) - 1) // 2
        already_analyzed_count = len(analyzed_pairs[name])
        
        if already_analyzed_count >= total_pairs:
            print(f"Skipping {name} - already analyzed")
            continue
            
        remaining_pairs = total_pairs - already_analyzed_count
        print(f"Analyzing {name} (pairwise) - {remaining_pairs}/{total_pairs} pairs remaining")
        
        # Initialize results for this analyzer if not present
        if name not in results:
            results[name] = []
        if f"{name}_errors" not in results:
            results[f"{name}_errors"] = []
            
        pair_count = 0
        new_results = []
        new_errors = []
        
        for i in range(len(components)):
            for j in range(i + 1, len(components)):  # Triangle logic - only upper triangle
                comp1, comp2 = components[i], components[j]
                pair_id = f"{comp1.id}:{comp2.id}"
                
                # Skip if already analyzed
                if pair_id in analyzed_pairs[name]:
                    continue
                    
                pair_count += 1
                print(f"  {pair_count}/{remaining_pairs}", end="")
                result = analyze_with_retry(analyzer(), client, 1, comp1, comp2)
                
                if result[0] is not None:
                    violations, usage = result
                    new_results.append({
                        "pair": pair_id,
                        "story1": comp1.original_text,
                        "story2": comp2.original_text,
                        "violations": [{"issue": v.issue, "suggestion": v.suggestion} for v in violations],
                        "violation_count": len(violations)
                    })
                    print(f" ✓ {len(violations)} violations")
                else:
                    new_errors.append({
                        "pair": pair_id,
                        "story1": comp1.original_text,
                        "story2": comp2.original_text,
                        "error": result[1]
                    })
                    print(f" ✗ Error: {result[1][:50]}...")
        
        # Add new results to existing results
        results[name].extend(new_results)
        results[f"{name}_errors"].extend(new_errors)
        
        # Save progress after each analyzer
        save_results(results)
    
    # Create separate CSV per criteria
    for name, analyzer_results in results.items():
        # Skip error entries for CSV generation
        if name.endswith("_errors"):
            continue
            
        with open(f"lucassen_{name}_violations.csv", 'w', newline='') as f:
            if name in individual_analyzers:
                f.write("User Story,Violations\n")
                for result in analyzer_results:
                    violations = ", ".join([v["issue"] for v in result["violations"]])
                    story = result["story"].replace('"', '""')
                    f.write(f'"{story}","{violations}"\n')
            else:  # Set analyzers
                f.write("Story1,Story2,Violations\n")
                for result in analyzer_results:
                    violations = ", ".join([v["issue"] for v in result["violations"]])
                    story1 = result["story1"].replace('"', '""')
                    story2 = result["story2"].replace('"', '""')
                    f.write(f'"{story1}","{story2}","{violations}"\n')
    
    print(f"Done - {len([k for k in results.keys() if not k.endswith('_errors')])} criteria analyzed")
    
    # Print error summary
    error_count = sum([len(v) for k, v in results.items() if k.endswith('_errors')])
    if error_count > 0:
        print(f"Total errors: {error_count}")

if __name__ == "__main__":
    main()