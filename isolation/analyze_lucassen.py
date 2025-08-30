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
from llm_qus_analyzer.individual.well_form import WellFormAnalyzer
from llm_qus_analyzer.set.uniform import UniformAnalyzer

def create_component(chunk_data, story_id):
    """Create QUSComponent from chunk data."""
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

def analyze_with_retry(analyzer_class, client, model_idx, *args, max_retries=5):
    """
    Analyze with exponential backoff retry logic for HTTP errors.
    
    Args:
        analyzer_class: The analyzer class (AtomicAnalyzer, MinimalAnalyzer, etc.)
        client: The LLMClient instance
        model_idx: Index of the model to use
        *args: Arguments to pass to the analyzer method
        max_retries: Maximum number of retry attempts
    
    Returns:
        tuple: (result, None) if successful, (None, error_message) if failed
    """
    base_delay = 1  # Start with 1 second delay
    
    for attempt in range(max_retries):
        try:
            # Use class method run for all analyzers
            result = analyzer_class.run(client, model_idx, *args)
            return result, None
        except json.JSONDecodeError as e:
            error_msg = str(e)
            print(f"\n  JSON Decode Error:")
            print(f"    Error: {error_msg}")
            print(f"    Position: line {e.lineno} column {e.colno} (char {e.pos})")
            return None, f"JSON parsing error: {error_msg}"
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a rate limit or HTTP error that we should retry
            if any(http_error in error_msg.lower() for http_error in ['http', '429', '500', '502', '503', '504', 'timeout', 'connection']):
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f" ⚠ HTTP error: {error_msg} Retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                    continue
            
            # For all other errors or if we've exhausted retries, return the error
            return None, error_msg
    
    # If we've exhausted all retries
    return None, f"Failed after {max_retries} attempts"



def load_existing_results(output_dir):
    """Load existing results from analysis.json if it exists."""
    try:
        results_file = output_dir / "analysis.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                content = f.read()
                if not content.strip():
                    print(f"Warning: {results_file} is empty")
                    return {}
                return json.loads(content)
        return {}
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Could not load existing results due to JSON error: {e}")
        print("Starting with fresh results...")
        return {}

def save_results(results, output_dir):
    """Save results to analysis.json."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "analysis.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Error saving results: {e}")

def create_violations_table(analyzer_results, individual_analyzers, analyzer_name, output_dir):
    """Create a CSV table with user stories and their violations."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_file = output_dir / f"{analyzer_name}_violations.csv"
        
        with open(csv_file, 'w', newline='') as f:
            if analyzer_name in individual_analyzers:
                # Individual analyzer format
                f.write("User Story,Violations\n")
                for result in analyzer_results:
                    # Convert violations list to a string format
                    violations_str = "; ".join([f"{v['issue']}" for v in result["violations"]])
                    story = result["story"].replace('"', '""')
                    f.write(f'"{story}","{violations_str}"\n')
            else:
                # Set analyzer format
                f.write("Story1,Story2,Violations\n")
                for result in analyzer_results:
                    # Convert violations list to a string format
                    violations_str = "; ".join([f"{v['issue']}" for v in result["violations"]])
                    story1 = result["story1"].replace('"', '""')
                    story2 = result["story2"].replace('"', '""')
                    f.write(f'"{story1}","{story2}","{violations_str}"\n')
        print(f"Created violations table: {csv_file}")
    except Exception as e:
        print(f"Error creating violations table: {e}")

def analyze_model(settings, model_idx, model_name, chunks):
    """Analyze chunks using a specific model."""
    print(f"\nAnalyzing with model: {model_name}")
    
    # Create output directory for this model within lucassen
    output_dir = Path("lucassen") / model_name.replace("/", "_").replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    client = LLMClient(settings)
    individual_analyzers = {"atomic": AtomicAnalyzer, "minimal": MinimalAnalyzer, "well-formed": WellFormAnalyzer}
    set_analyzers = {"uniform": UniformAnalyzer}
    
    # Load existing results to avoid re-analyzing
    results = load_existing_results(output_dir)
    
    # Individual analysis
    for name, analyzer in individual_analyzers.items():
        # Skip if already analyzed all chunks
        if name in results and len(results[name]) >= len(chunks):
            print(f"  Skipping {name} - already analyzed")
            # Create violations table if it doesn't exist
            violations_table_path = output_dir / f"{name}_violations.csv"
            if not violations_table_path.exists():
                create_violations_table(results[name], individual_analyzers, name, output_dir)
            continue
            
        print(f"  Analyzing {name}")
        analyzer_results = results.get(name, [])
        errors = results.get(f"{name}_errors", [])
        
        # Create a set of already analyzed story IDs
        analyzed_story_ids = set(result["story_id"] for result in analyzer_results)
        analyzed_story_ids.update(result["story_id"] for result in errors)
        
        # Process only unanalyzed chunks
        unanalyzed_chunks = [(chunk_id, chunk_data) for chunk_id, chunk_data in chunks.items() 
                            if chunk_id not in analyzed_story_ids]
        
        if not unanalyzed_chunks:
            print(f"  Skipping {name} - all chunks already analyzed")
            continue
            
        print(f"  Processing {len(unanalyzed_chunks)} unanalyzed chunks out of {len(chunks)} total")
        
        for i, (chunk_id, chunk_data) in enumerate(unanalyzed_chunks):
            print(f"    {i+1}/{len(unanalyzed_chunks)}", end="")
            component = create_component(chunk_data, chunk_id)
            result = analyze_with_retry(analyzer, client, model_idx, component)
            
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
                print(f" ✗ Error: {result[1]}")
        
        results[name] = analyzer_results
        if errors:
            results[f"{name}_errors"] = errors
            
        # Save progress after each analyzer
        save_results(results, output_dir)
        
        # Create violations table
        create_violations_table(analyzer_results, individual_analyzers, name, output_dir)
    
    # Set analysis - Uniform only
    components = [create_component(chunk_data, chunk_id) for chunk_id, chunk_data in chunks.items()]
    
    for name, analyzer in set_analyzers.items():
        print(f"  Analyzing {name} (template uniformity)")
        
        # Check if already analyzed
        if name in results and len(results[name]) > 0:
            print(f"  Skipping {name} - already analyzed")
            violations_table_path = output_dir / f"{name}_violations.csv"
            if not violations_table_path.exists():
                if name not in results:
                    results[name] = []
                create_violations_table(results[name], individual_analyzers, name, output_dir)
            continue
        
        # Process uniform analysis
        try:
            uniform_results = analyze_with_retry(analyzer, client, model_idx, components)
            
            if uniform_results[0] is not None:
                # UniformAnalyzer returns list of (violations, usage) tuples directly
                component_results = uniform_results[0]
                uniform_violations = []
                
                # Process each component's result
                for i, (violations, usage_dict) in enumerate(component_results):
                    if violations:  # Only add if there are violations
                        uniform_violations.append({
                            "story_id": components[i].id,
                            "story": components[i].original_text,
                            "violations": [{"issue": v.issue, "suggestion": v.suggestion} for v in violations],
                            "violation_count": len(violations)
                        })
                
                results[name] = uniform_violations
                print(f"  ✓ {len(uniform_violations)} stories with uniformity violations")
            else:
                results[f"{name}_errors"] = [{
                    "error": uniform_results[1],
                    "total_stories": len(components)
                }]
                print(f"  ✗ Error: {uniform_results[1]}")
                
            save_results(results, output_dir)
            create_violations_table(results[name], individual_analyzers, name, output_dir)
            
        except Exception as e:
            error_msg = str(e)
            results[f"{name}_errors"] = [{
                "error": error_msg,
                "total_stories": len(components)
            }]
            print(f"  ✗ Error: {error_msg}")
            save_results(results, output_dir)
    
    print(f"  Done with model {model_name}")
    
    # Print error summary
    error_count = sum([len(v) for k, v in results.items() if k.endswith('_errors')])
    if error_count > 0:
        print(f"  Total errors for {model_name}: {error_count}")
    
    return results

def load_chunks_from_lucassen():
    """Load chunks from existing lucassen directory structure."""
    lucassen_dir = Path("lucassen")
    if not lucassen_dir.exists():
        raise FileNotFoundError("lucassen directory not found")
    
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
    
    raise FileNotFoundError("No valid chunks.json found in lucassen directory")

def main():
    settings = Settings()
    settings.configure_paths_and_load("../.env", "models.yaml")
    client = LLMClient(settings)
    
    chunks = load_chunks_from_lucassen()
    
    # Analyze with each model
    all_results = {}
    for model_idx, model_info in enumerate(client._LLMClient__model_info):
        model_name = model_info.name
        results = analyze_model(settings, model_idx, model_name, chunks)
        all_results[model_name] = results
    
    # Create a combined summary file
    summary = {}
    for model_name, results in all_results.items():
        # Count total violations across all analyzers
        total_violations = 0
        total_errors = 0
        
        for key, value in results.items():
            if key.endswith('_errors'):
                total_errors += len(value)
            elif isinstance(value, list):
                total_violations += len(value)
        
        summary[model_name] = {
            "total_violations": total_violations,
            "total_errors": total_errors
        }
    
    with open("lucassen/analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nAll models analyzed successfully!")
    print("Summary:")
    for model_name, stats in summary.items():
        print(f"  {model_name}: {stats['total_violations']} violations, {stats['total_errors']} errors")

if __name__ == "__main__":
    main()