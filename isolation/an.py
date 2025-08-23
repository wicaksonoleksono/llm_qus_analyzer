#!/usr/bin/env python3
"""
Per-Criteria Analyzer - Analyzes pre-chunked stories by PT category
Uses cached chunks and saves results to separate JSON files per criteria.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

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
from llm_qus_analyzer.individual.unambigous import UnambiguousAnalyzer

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
    "unambiguous": UnambiguousAnalyzer,
}

SET_CATEGORIES = ["conflict-free", "complete", "independent", "unique"]
INDIVIDUAL_CATEGORIES = ["conceptually-sound", "problem-oriented", "full-sentence", "estimatable", "unambiguous"]


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


def load_existing_results(file_path: str) -> Dict:
    """Load existing analysis results from JSON file if it exists."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


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
        id=story_id or chunk_data.get("id"),
        original_text=chunk_data.get("original_story")
    )


def _format_violation_parts(violation):
    """Format violation parts based on violation type."""
    from llm_qus_analyzer.type import PairwiseViolation, Violation, FullSetViolation
    
    if isinstance(violation, PairwiseViolation):
        return {
            "first_parts": list(violation.first_parts),
            "second_parts": list(violation.second_parts),
            "first_id": violation.first_id,
            "second_id": violation.second_id,
            "violation_type": "pairwise"
        }
    elif isinstance(violation, Violation):
        return {
            "parts": list(violation.parts),
            "violation_type": "individual"
        }
    elif isinstance(violation, FullSetViolation):
        return {
            "story_ids": violation.story_ids,
            "parts_per_story": [list(parts) for parts in violation.parts_per_story],
            "violation_type": "fullset"
        }
    else:
        # Fallback for unknown violation types
        return {
            "parts": list(getattr(violation, 'parts', [])),
            "violation_type": "unknown"
        }


def _evaluate_violation_match(expected_violation: str, detected_violations: List) -> bool:
    """
    Evaluate if detected violations match expected violations.
    
    Returns True for successful prediction (TP or TN), False for failure (FP or FN).
    """
    # Parse expected violation
    is_expected_none = expected_violation.startswith("None (Valid.")
    
    # Check if any violations were detected
    has_detected_violations = len(detected_violations) > 0
    
    if is_expected_none:
        # True Negative: Expected no violations, detected no violations
        return not has_detected_violations
    else:
        # Expected specific violation type - any detected violation counts as success
        # True Positive: Expected violation and found any violation
        return has_detected_violations


async def safe_analyzer_call_async(analyzer_func, *args, max_retries=5, **kwargs):
    """Safely call analyzer with error handling and exponential backoff retry logic."""
    import json
    
    for attempt in range(max_retries):
        try:
            return analyzer_func(*args, **kwargs)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                continue
            else:
                print("Max retries reached for JSON parsing. Raising error.")
                raise e
        except Exception as e:
            error_msg = str(e).lower()
            if "503" in error_msg or "service unavailable" in error_msg:
                wait_time = min(10 * (2 ** attempt), 120)  # Exponential backoff, max 2 minutes
                print(f"API unavailable (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print("Max retries reached for API unavailable error. Raising error.")
                    raise e
            elif "429" in error_msg or "rate limit" in error_msg:
                wait_time = min(5 * (2 ** attempt), 60)  # Exponential backoff, max 1 minute
                print(f"Rate limited (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print("Max retries reached for rate limit error. Raising error.")
                    raise e
            else:
                print(f"Analyzer error: {e}")
                raise e
    raise Exception("Max retries reached")


async def analyze_criteria_set(client: LLMClient, test_data: List[Dict], 
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
        
        for chunk_id, chunk_data in chunks.items():
            if chunk_data.get("original_story") == stories[0]:
                chunk1 = chunk_data
            elif chunk_data.get("original_story") == stories[1]:
                chunk2 = chunk_data
        
        if not chunk1 or not chunk2:
            print(f"    Missing chunks, skipping...")
            continue
            
        comp1 = create_component_from_chunk(chunk1, f"story_{i+1}_1")
        comp2 = create_component_from_chunk(chunk2, f"story_{i+1}_2")
        
        # Run analysis (set analyzers use class method analyze_pairwise)  
        # Direct call to avoid parameter mismatch with class methods
        violations, usage_dict = await safe_analyzer_call_async(
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
                    **_format_violation_parts(v)
                } for v in violations
            ],
            "has_violations": len(violations) > 0,
            "violation_count": len(violations),
            "usage_stats": usage_dict
        }
        
        results["pairs"].append(pair_result)
        await asyncio.sleep(0.5)  # Rate limiting
    
    # Calculate summary
    results["summary"]["total_pairs"] = len(results["pairs"])
    results["summary"]["pairs_with_violations"] = sum(1 for p in results["pairs"] if p["has_violations"])
    results["summary"]["total_violations"] = sum(p["violation_count"] for p in results["pairs"])
    
    if results["summary"]["total_pairs"] > 0:
        results["summary"]["success_rate"] = results["summary"]["pairs_with_violations"] / results["summary"]["total_pairs"]
    
    return results



async def analyze_criteria_set_fullset(client: LLMClient, test_data: List[Dict], 
                                 chunks: Dict, criteria: str) -> Dict:
    """Analyze set criteria using full-set analysis only."""
    
    analyzer = ANALYZER_MAP[criteria]
    results = {
        "criteria": criteria,
        "analyzer_type": "set_fullset",
        "analyzer_name": analyzer.__name__,
        "violations": [],
        "summary": {
            "total_violations": 0,
            "has_violations": False
        }
    }
    
    # Build component list from all unique stories
    all_components = []
    story_to_component = {}
    story_id_mapping = {}  # Maps LLM story position (1-indexed) to original story text
    
    # Filter test data for this criteria and collect unique stories  
    criteria_items = [item for item in test_data 
                     if item.get("pt") == criteria and "stories" in item]
    
    print(f"Analyzing {len(criteria_items)} pairs for '{criteria}' criteria (fullset mode)")
    
    for i, item in enumerate(criteria_items):
        if len(item["stories"]) != 2:
            continue
            
        stories = item["stories"]
        
        for story in stories:
            if story not in story_to_component:
                # Find chunk by original story text
                chunk_data = None
                for chunk_id, chunk in chunks.items():
                    if chunk.get("original_story") == story:
                        chunk_data = chunk
                        break
                
                if chunk_data:
                    component = create_component_from_chunk(chunk_data, f"story_{len(all_components)+1}")
                    story_to_component[story] = component
                    all_components.append(component)
                    # Map LLM story number (1-indexed) to original story text
                    story_id_mapping[len(all_components)] = story
    
    print(f"  Built component set of {len(all_components)} unique stories")
    print(f"  Story ID mapping: {story_id_mapping}")
    
    # Run fullset analysis if we have enough components
    if len(all_components) >= 2:
        fullset_violations, usage_dict = await safe_analyzer_call_async(
            analyzer.run, client, 0, all_components, mode="fullset"
        )
        
        # Process results with story mapping  
        processed_violations = []
        for v in fullset_violations:
            # Map LLM story IDs back to original stories
            mapped_stories = []
            for story_id in v.story_ids:
                if story_id in story_id_mapping:
                    mapped_stories.append(story_id_mapping[story_id])
                else:
                    mapped_stories.append(f"story_{story_id}")  # fallback
            
            processed_violations.append({
                "issue": v.issue,
                "suggestion": v.suggestion,
                "story_ids": v.story_ids,  # Original LLM IDs
                "original_stories": mapped_stories,  # Mapped back to actual story text
                "parts_per_story": [list(parts) for parts in v.parts_per_story],
                "violation_type": "fullset"
            })
        
        results["violations"] = processed_violations
        results["story_mapping"] = story_id_mapping  # Include mapping for validation
        results["summary"]["total_violations"] = len(fullset_violations)
        results["summary"]["has_violations"] = len(fullset_violations) > 0
        results["usage_stats"] = usage_dict
    
    return results


async def analyze_criteria_individual(client: LLMClient, test_data: List[Dict], 
                               chunks: Dict, criteria: str, existing_results: Dict = None, revise_only: bool = False) -> Dict:
    """Analyze individual criteria (single story analysis), skipping already analyzed stories."""
    
    analyzer = ANALYZER_MAP[criteria]
    results = {
        "criteria": criteria,
        "analyzer_type": "individual", 
        "analyzer_name": analyzer.__name__,
        "stories": existing_results.get("stories", []) if existing_results else [],
        "summary": {
            "total_stories": 0,
            "stories_with_violations": 0,
            "total_violations": 0,
            "success_rate": 0.0
        }
    }
    
    # Handle revision mode vs normal mode
    revision_tracker = {}  # Track original failed stories for reporting
    if revise_only:
        # In revision mode, only process stories with is_success == false
        failed_stories = []
        if existing_results and "stories" in existing_results:
            failed_stories = [story for story in existing_results["stories"] 
                            if story.get("is_success") == False]
        
        print(f"Revision mode: Found {len(failed_stories)} failed stories to re-analyze")
        
        # Track original failed stories for reporting
        for story in failed_stories:
            revision_tracker[story["story_id"]] = {
                "story_text": story["story"][:50] + "..." if len(story["story"]) > 50 else story["story"],
                "original_success": False,
                "new_success": None  # Will be updated after re-analysis
            }
        
        # Convert failed stories back to test data format for processing
        revision_items = []
        for story_result in failed_stories:
            revision_items.append({
                "story": story_result["story"],
                "violation": story_result["expected_violation"],
                "pt": criteria
            })
        
        items_to_process = revision_items
    else:
        # Normal mode: skip already analyzed stories
        already_analyzed = set()
        if existing_results and "stories" in existing_results:
            already_analyzed = {story["story"] for story in existing_results["stories"]}
        
        # Filter test data for this criteria
        criteria_items = [item for item in test_data 
                         if item.get("pt") == criteria and "story" in item]
        
        # Filter out already analyzed stories
        items_to_process = [item for item in criteria_items if item["story"] not in already_analyzed]
        
        print(f"Analyzing {len(criteria_items)} stories for '{criteria}' criteria ({len(items_to_process)} new, {len(already_analyzed)} existing)")
    
    for i, item in enumerate(items_to_process):
        story = item["story"]
        expected_violation = item.get("violation", "")
        
        print(f"  Story {i+1}/{len(items_to_process)}: {expected_violation}")
        
        # Find chunk by original story text
        chunk_data = None
        
        for chunk_id, chunk in chunks.items():
            if chunk.get("original_story") == story:
                chunk_data = chunk
                break
        
        if not chunk_data:
            print(f"    Missing chunk, skipping...")
            continue
        
        # Handle story ID assignment
        if revise_only:
            # In revision mode, find the original story ID from existing results
            original_story = next((s for s in existing_results["stories"] 
                                 if s["story"] == story and s.get("is_success") == False), None)
            story_id_num = original_story["story_id"] if original_story else i + 1
        else:
            # Normal mode: use story ID that accounts for existing stories
            story_id_num = len(already_analyzed) + i + 1
        
        component = create_component_from_chunk(chunk_data, f"story_{story_id_num}")
        
        # Run analysis (individual analyzers use class method run)
        # Direct call to avoid parameter mismatch with class methods
        violations, usage_dict = await safe_analyzer_call_async(
            analyzer.run, client, 0, component
        )
        
        # Evaluate success based on expected vs detected violations
        is_success = _evaluate_violation_match(expected_violation, violations)
        
        # Process results
        story_result = {
            "story_id": story_id_num,
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
                    **_format_violation_parts(v)
                } for v in violations
            ],
            "has_violations": len(violations) > 0,
            "violation_count": len(violations),
            "is_success": is_success,
            "usage_stats": usage_dict
        }
        
        
        if revise_only:
            # Update revision tracker with new success status
            if story_id_num in revision_tracker:
                revision_tracker[story_id_num]["new_success"] = is_success
            
            # Replace the existing story in-place
            for idx, existing_story in enumerate(results["stories"]):
                if existing_story["story_id"] == story_id_num:
                    results["stories"][idx] = story_result
                    break
        else:
            # Normal mode: append new story
            results["stories"].append(story_result)
        await asyncio.sleep(0.5)  # Rate limiting
    
    # Calculate summary
    results["summary"]["total_stories"] = len(results["stories"])
    results["summary"]["stories_with_violations"] = sum(1 for s in results["stories"] if s["has_violations"])
    results["summary"]["total_violations"] = sum(s["violation_count"] for s in results["stories"])
    results["summary"]["successful_predictions"] = sum(1 for s in results["stories"] if s["is_success"])
    
    if results["summary"]["total_stories"] > 0:
        results["summary"]["success_rate"] = results["summary"]["successful_predictions"] / results["summary"]["total_stories"]
    
    # Add revision report if in revision mode
    if revise_only and revision_tracker:
        resolved_stories = [sid for sid, info in revision_tracker.items() 
                           if info["new_success"] == True]
        still_failing = [sid for sid, info in revision_tracker.items() 
                        if info["new_success"] == False]
        
        print(f"\n{'='*60}")
        print(f"REVISION REPORT")
        print(f"{'='*60}")
        print(f"✅ Resolved: {len(resolved_stories)}/{len(revision_tracker)} stories")
        print(f"❌ Still failing: {len(still_failing)}/{len(revision_tracker)} stories")
        
        if resolved_stories:
            print(f"\n🎉 RESOLVED STORIES:")
            for sid in resolved_stories:
                story_info = revision_tracker[sid]
                print(f"  • Story {sid}: {story_info['story_text']}")
        
        if still_failing:
            print(f"\n⚠️  STILL FAILING STORIES:")
            for sid in still_failing:
                story_info = revision_tracker[sid]
                print(f"  • Story {sid}: {story_info['story_text']}")
        
        results["revision_report"] = {
            "total_revised": len(revision_tracker),
            "resolved": len(resolved_stories),
            "still_failing": len(still_failing),
            "resolved_story_ids": resolved_stories,
            "still_failing_story_ids": still_failing
        }
    
    return results


async def main():
    """Main function for per-criteria analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze stories by specific criteria')
    parser.add_argument('--criteria', '-c', type=str, 
                       help='Specific criteria to analyze (e.g., complete, conflict-free)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available criteria')
    parser.add_argument('--set-only', action='store_true',
                       help='Only analyze set criteria (pairwise)')
    parser.add_argument('--individual-only', action='store_true', 
                       help='Only analyze individual criteria')
    parser.add_argument('--revise-only', '-r', action='store_true',
                       help='Only re-analyze stories with "is_success": false')
    parser.add_argument('--type', '-t', choices=['p', 'a'], default='p',
                       help='Analysis type for set criteria: p=pairwise, a=all (fullset)')
    
    args = parser.parse_args()
    
    # Configuration
    output_dir = Path(__file__).parent / "output"
    pre_chunked_dir = Path(__file__).parent / "pre-chunked"
    testdata_dir = Path(__file__).parent / "testdata" / "by-criteria"
    models_path = Path(__file__).parent.parent / "models.yaml"
    env_path = Path(__file__).parent.parent / ".env"
    
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
        
        # Get available criteria from pre-chunked files
        print("\nScanning for available criteria...")
        available_criteria = set()
        
        for chunk_file in pre_chunked_dir.glob("*_chunks.json"):
            criteria = chunk_file.stem.replace("_chunks", "")
            if criteria in ANALYZER_MAP:
                available_criteria.add(criteria)
        
        print(f"\nAvailable criteria: {sorted(available_criteria)}")
        
        # Handle --list flag
        if args.list:
            print("\nAll available criteria:")
            for criteria in sorted(available_criteria):
                criteria_type = "SET" if criteria in SET_CATEGORIES else "INDIVIDUAL"
                print(f"  {criteria} ({criteria_type})")
            return
        
        # Filter criteria based on flags
        criteria_to_analyze = available_criteria
        
        if args.criteria:
            if args.criteria not in available_criteria:
                print(f"\n❌ Error: '{args.criteria}' not found in available criteria")
                print(f"Available: {sorted(available_criteria)}")
                return
            criteria_to_analyze = {args.criteria}
            print(f"\n🎯 Analyzing single criteria: {args.criteria}")
        
        if args.set_only:
            criteria_to_analyze = {c for c in criteria_to_analyze if c in SET_CATEGORIES}
            print(f"\n📊 Analyzing SET criteria only: {sorted(criteria_to_analyze)}")
        
        if args.individual_only:
            criteria_to_analyze = {c for c in criteria_to_analyze if c not in SET_CATEGORIES}
            print(f"\n👤 Analyzing INDIVIDUAL criteria only: {sorted(criteria_to_analyze)}")
        
        if not criteria_to_analyze:
            print("\n❌ No criteria match the specified filters")
            return
        
        # Analyze each criteria
        all_results = {}
        
        for criteria in sorted(criteria_to_analyze):
            print(f"\n{'='*50}")
            print(f"ANALYZING CRITERIA: {criteria.upper()}")
            print(f"{'='*50}")
            
            # Load data for this specific criteria
            chunks_file = pre_chunked_dir / f"{criteria}_chunks.json"
            test_data_file = testdata_dir / f"{criteria}.json"
            
            if not chunks_file.exists():
                print(f"❌ Chunks file not found: {chunks_file}")
                print(f"Run: python chunk_by_criteria.py --criteria {criteria}")
                continue
            
            if not test_data_file.exists():
                print(f"❌ Test data file not found: {test_data_file}")
                continue
            
            print(f"📁 Loading chunks: {chunks_file}")
            print(f"📁 Loading test data: {test_data_file}")
            
            chunks = load_chunks(str(chunks_file))
            test_data = load_test_data(str(test_data_file))
            
            # Create filename with analysis mode suffix
            mode_suffix = ""
            if criteria in SET_CATEGORIES:
                if args.type == 'p':
                    mode_suffix = "-pair"
                elif args.type == 'a':
                    mode_suffix = "-free"
            
            # Load existing results if available
            criteria_file = output_dir / f"{criteria}{mode_suffix}_analysis.json"
            existing_results = load_existing_results(str(criteria_file))
            existing_count = len(existing_results.get("stories", [])) if "stories" in existing_results else 0
            
            print(f"Loaded {len(chunks)} chunks, {len(test_data)} test items, and {existing_count} existing results")
            
            # Validate revise-only mode
            if args.revise_only and existing_count == 0:
                print(f"❌ Revise-only mode requires existing results. No results found for '{criteria}'")
                continue
            
            if criteria in SET_CATEGORIES:
                if args.type == 'p':
                    results = await analyze_criteria_set(client, test_data, chunks, criteria)
                elif args.type == 'a':
                    results = await analyze_criteria_set_fullset(client, test_data, chunks, criteria)
            else:
                results = await analyze_criteria_individual(client, test_data, chunks, criteria, existing_results, args.revise_only)
            
            all_results[criteria] = results
            
            # Save individual criteria results
            criteria_file = output_dir / f"{criteria}{mode_suffix}_analysis.json"
            with open(criteria_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n✅ Results saved: {criteria_file}")
            
            if results.get("analyzer_type") == "set_fullset":
                # Fullset mode reporting
                total_violations = results["summary"]["total_violations"]
                has_violations = results["summary"]["has_violations"]
                print(f"📦 Full-set Analysis Results:")
                print(f"   Total violations: {total_violations}")
                print(f"   Has violations: {has_violations}")
            elif "stories" in results:
                # Individual mode reporting
                new_count = len(results["stories"]) - existing_count
                print(f"📊 Processed {len(results['stories'])} total stories ({new_count} new, {existing_count} existing)")
                print(f"Summary: {results['summary']}")
            else:
                # Set mode reporting
                total_pairs = results.get("summary", {}).get("total_pairs", 0)
                pairs_with_violations = results.get("summary", {}).get("pairs_with_violations", 0)
                print(f"📊 Processed {total_pairs} pairs ({pairs_with_violations} with violations)")
                print(f"Summary: {results['summary']}")
        
        print(f"\n{'='*60}")
        print(f"PER-CRITERIA ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        print(f"✅ Individual results: {output_dir}/<criteria>_analysis.json")
        print(f"✅ Analyzed {len(available_criteria)} criteria")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())