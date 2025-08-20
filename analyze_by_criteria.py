#!/usr/bin/env python3
"""
Per-Criteria Analyzer - Analyzes pre-chunked stories by PT category
Uses cached chunks and saves results to separate JSON files per criteria.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent))

from llm_qus_analyzer.client import LLMClient
from llm_qus_analyzer.settings import Settings

# Set analyzers
from llm_qus_analyzer.set.conflict_free import ConflictFreeAnalyzer
from llm_qus_analyzer.set.complete import CompleteAnalyzer
from llm_qus_analyzer.set.independent import IndependentAnalyzer
from llm_qus_analyzer.set.unique import UniqueAnalyzer
from llm_qus_analyzer.set.uniform import UniformAnalyzer

# Individual analyzers
from llm_qus_analyzer.individual.conceptually import ConceptuallySoundAnalyzer
from llm_qus_analyzer.individual.problem_oriented import ProblemOrientedAnalyzer
from llm_qus_analyzer.individual.full_sentence import FullSentenceAnalyzer
from llm_qus_analyzer.individual.estimatable import EstimatableAnalyzer

# PT category mappings to analyzers
ANALYZER_MAP = {
    # Set analyzers
    "conflict-free": ConflictFreeAnalyzer,
    "complete": CompleteAnalyzer,
    "independent": IndependentAnalyzer,
    "unique": UniqueAnalyzer,
    
    # Individual analyzers  
    "conceptually-sound": ConceptuallySoundAnalyzer,
    "problem-oriented": ProblemOrientedAnalyzer,
    "full-sentence": FullSentenceAnalyzer,
    "estimatable": EstimatableAnalyzer,
}

SET_CATEGORIES = ["conflict-free", "complete", "independent", "unique"]
INDIVIDUAL_CATEGORIES = ["conceptually-sound", "problem-oriented", "full-sentence", "estimatable"]


def load_chunks(file_path: str) -> Dict:
    """Load chunked components from file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Chunks file not found: {file_path}")
        print("Run chunking phase first!")
        return {}


def load_test_data(file_path: str) -> List[Dict]:
    """Load test data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def create_component_from_chunk(chunk_data: Dict, story_id: str = None):
    """Create QUSComponent from cached chunk data."""
    from llm_qus_analyzer.chunker.models import QUSComponent
    from llm_qus_analyzer.chunker.parser import Template, TemplateParser
    
    # Check if we have stored template data
    if "template_data" in chunk_data:
        # Use stored template data for proper reconstruction
        template_data = chunk_data["template_data"]
        template = Template(
            text=template_data["text"],
            chunk=template_data["chunk"],
            tail=template_data["tail"],
            order=template_data["order"]
        )
    else:
        # Fallback: create template using parser (for old chunks without template_data)
        TemplateParser.prepare()
        template = TemplateParser.parse(
            chunk_data["text"], 
            chunk_data["role"], 
            chunk_data["means"], 
            chunk_data["ends"]
        )
    
    return QUSComponent(
        text=chunk_data["text"],
        role=chunk_data["role"],
        means=chunk_data["means"],
        ends=chunk_data["ends"],
        template=template,
        id=story_id or chunk_data.get("id")
    )


def safe_analyzer_call(analyzer_func, *args, max_retries=3, **kwargs):
    """Safely call analyzer with error handling and retry logic."""
    for attempt in range(max_retries):
        try:
            return analyzer_func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if "503" in error_msg or "service unavailable" in error_msg:
                print(f"API unavailable (attempt {attempt + 1}): Retrying...")
                if attempt < max_retries - 1:
                    time.sleep(10)
                    continue
            print(f"Analyzer error: {e}")
            return [], {}
    return [], {}


def analyze_criteria_set(client: LLMClient, test_data: List[Dict], 
                        chunks: Dict, criteria: str) -> Dict:
    """Analyze set criteria (pairwise analysis)."""
    
    analyzer = ANALYZER_MAP[criteria]
    results = {
        "criteria": criteria,
        "analyzer_type": "set",
        "analyzer_name": analyzer.__name__,
        "pairs": [],
        "summary": {
            "total_pairs": 0,
            "pairs_with_violations": 0,
            "total_violations": 0,
            "success_rate": 0.0
        }
    }
    
    # Filter test data for this criteria
    criteria_items = [item for item in test_data 
                     if item.get("pt") == criteria and "stories" in item]
    
    print(f"Analyzing {len(criteria_items)} pairs for '{criteria}' criteria")
    
    for i, item in enumerate(criteria_items):
        if len(item["stories"]) != 2:
            continue
            
        stories = item["stories"]
        expected_violation = item.get("violation", "")
        
        print(f"  Pair {i+1}/{len(criteria_items)}: {expected_violation}")
        
        # Find chunks by original story text
        chunk1 = None
        chunk2 = None
        
        for chunk_hash, chunk_data in chunks.items():
            if chunk_data.get("original_story") == stories[0]:
                chunk1 = chunk_data
            elif chunk_data.get("original_story") == stories[1]:
                chunk2 = chunk_data
        
        if not chunk1 or not chunk2:
            print(f"    Missing chunks, skipping...")
            continue
            
        comp1 = create_component_from_chunk(chunk1, f"story_{i+1}_1")
        comp2 = create_component_from_chunk(chunk2, f"story_{i+1}_2")
        
        # Run analysis
        violations, usage_dict = safe_analyzer_call(
            analyzer.analyze_pairwise, client, 0, comp1, comp2
        )
        
        # Process results
        pair_result = {
            "pair_id": i + 1,
            "expected_violation": expected_violation,
            "stories": stories,
            "components": {
                "story_1": {
                    "role": comp1.role,
                    "means": comp1.means, 
                    "ends": comp1.ends
                },
                "story_2": {
                    "role": comp2.role,
                    "means": comp2.means,
                    "ends": comp2.ends
                }
            },
            "detected_violations": [
                {
                    "issue": v.issue,
                    "suggestion": v.suggestion,
                    "first_parts": list(getattr(v, 'first_parts', v.parts)),
                    "second_parts": list(getattr(v, 'second_parts', v.parts))
                } for v in violations
            ],
            "has_violations": len(violations) > 0,
            "violation_count": len(violations),
            "usage_stats": usage_dict
        }
        
        results["pairs"].append(pair_result)
        time.sleep(1)  # Rate limiting
    
    # Calculate summary
    results["summary"]["total_pairs"] = len(results["pairs"])
    results["summary"]["pairs_with_violations"] = sum(1 for p in results["pairs"] if p["has_violations"])
    results["summary"]["total_violations"] = sum(p["violation_count"] for p in results["pairs"])
    
    if results["summary"]["total_pairs"] > 0:
        results["summary"]["success_rate"] = results["summary"]["pairs_with_violations"] / results["summary"]["total_pairs"]
    
    return results


def analyze_criteria_individual(client: LLMClient, test_data: List[Dict], 
                               chunks: Dict, criteria: str) -> Dict:
    """Analyze individual criteria (single story analysis)."""
    
    analyzer = ANALYZER_MAP[criteria]
    results = {
        "criteria": criteria,
        "analyzer_type": "individual", 
        "analyzer_name": analyzer.__name__,
        "stories": [],
        "summary": {
            "total_stories": 0,
            "stories_with_violations": 0,
            "total_violations": 0,
            "success_rate": 0.0
        }
    }
    
    # Filter test data for this criteria
    criteria_items = [item for item in test_data 
                     if item.get("pt") == criteria and "story" in item]
    
    print(f"Analyzing {len(criteria_items)} stories for '{criteria}' criteria")
    
    for i, item in enumerate(criteria_items):
        story = item["story"]
        expected_violation = item.get("violation", "")
        
        print(f"  Story {i+1}/{len(criteria_items)}: {expected_violation}")
        
        # Find chunk by original story text
        chunk_data = None
        
        for chunk_hash, chunk in chunks.items():
            if chunk.get("original_story") == story:
                chunk_data = chunk
                break
        
        if not chunk_data:
            print(f"    Missing chunk, skipping...")
            continue
        
        component = create_component_from_chunk(chunk_data, f"story_{i+1}")
        
        # Run analysis
        violations, usage_dict = safe_analyzer_call(
            analyzer.run, client, 0, component
        )
        
        # Process results
        story_result = {
            "story_id": i + 1,
            "expected_violation": expected_violation,
            "story": story,
            "component": {
                "role": component.role,
                "means": component.means,
                "ends": component.ends
            },
            "detected_violations": [
                {
                    "issue": v.issue,
                    "suggestion": v.suggestion,
                    "parts": list(v.parts) if hasattr(v, 'parts') else []
                } for v in violations
            ],
            "has_violations": len(violations) > 0,
            "violation_count": len(violations),
            "usage_stats": usage_dict
        }
        
        results["stories"].append(story_result)
        time.sleep(1)  # Rate limiting
    
    # Calculate summary
    results["summary"]["total_stories"] = len(results["stories"])
    results["summary"]["stories_with_violations"] = sum(1 for s in results["stories"] if s["has_violations"])
    results["summary"]["total_violations"] = sum(s["violation_count"] for s in results["stories"])
    
    if results["summary"]["total_stories"] > 0:
        results["summary"]["success_rate"] = results["summary"]["stories_with_violations"] / results["summary"]["total_stories"]
    
    return results


def main():
    """Main function for per-criteria analysis."""
    
    # Configuration
    test_data_path = Path(__file__).parent / "testdata" / "semantic_pragmatic_test.json"
    chunks_file = Path(__file__).parent / "story_chunks.json"
    output_dir = Path(__file__).parent / "output"
    models_path = Path(__file__).parent / "models.yaml"
    env_path = Path(__file__).parent / ".env"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Setup
        settings = Settings()
        settings.configure_paths_and_load(env_path if env_path.exists() else None, models_path)
        
        client = LLMClient(settings)
        
        print(f"Per-Criteria Analyzer")
        print(f"Model: {client.names[0]}")
        print(f"Output directory: {output_dir}")
        
        # Load data
        print("\nLoading data...")
        chunks = load_chunks(chunks_file)
        if not chunks:
            print("No chunks found! Run chunking phase first.")
            return
            
        test_data = load_test_data(test_data_path)
        
        print(f"Loaded {len(chunks)} chunks and {len(test_data)} test items")
        
        # Get available criteria from test data
        available_criteria = set()
        for item in test_data:
            pt = item.get("pt")
            if pt and pt in ANALYZER_MAP:
                available_criteria.add(pt)
        
        print(f"\nAvailable criteria: {sorted(available_criteria)}")
        
        # Analyze each criteria
        all_results = {}
        
        for criteria in sorted(available_criteria):
            print(f"\n{'='*50}")
            print(f"ANALYZING CRITERIA: {criteria.upper()}")
            print(f"{'='*50}")
            
            if criteria in SET_CATEGORIES:
                results = analyze_criteria_set(client, test_data, chunks, criteria)
            else:
                results = analyze_criteria_individual(client, test_data, chunks, criteria)
            
            all_results[criteria] = results
            
            # Save individual criteria results
            criteria_file = output_dir / f"{criteria}_analysis.json"
            with open(criteria_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n✅ Results saved: {criteria_file}")
            print(f"Summary: {results['summary']}")
        
        # Save combined results
        summary_file = output_dir / "all_criteria_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print(f"PER-CRITERIA ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        print(f"✅ Individual results: {output_dir}/<criteria>_analysis.json")
        print(f"✅ Combined summary: {summary_file}")
        print(f"✅ Analyzed {len(available_criteria)} criteria")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()